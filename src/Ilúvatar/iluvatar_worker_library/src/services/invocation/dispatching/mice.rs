use crate::services::invocation::dispatching::queueing_dispatcher::DispatchPolicy;
use crate::services::invocation::dispatching::{QueueMap, NO_ESTIMATE};
// removed: use crate::services::invocation::queueing::DeviceQueue;
use crate::services::registration::RegisteredFunction;
use crate::worker_api::config::InvocationConfig;
use anyhow::Result;
use iluvatar_library::char_map::{Chars, WorkerCharMap};
use iluvatar_library::clock::{get_global_clock, Clock};
use iluvatar_library::transaction::TransactionId;
use iluvatar_library::types::Compute;
use parking_lot::Mutex;
use std::sync::Arc;
use time::OffsetDateTime;
use tracing::{info, warn}; // removed debug

/// Internal state for the MICE (Machine-Learning ICE) policy.
#[derive(Debug)]
struct MiceState {
    /// Dispatch threshold (jobs with size < tau go to the GPU).
    tau: f64,
    /// Cumulative workload dispatched to the GPU during the current epoch.
    gpu_work: f64,
    /// Cumulative workload dispatched to the CPU during the current epoch.
    cpu_work: f64,
    /// Number of invocations dispatched during the current epoch.
    count: u64,
    /// Timestamp marking the start of the current epoch.
    last_update: OffsetDateTime,
}

/// Mice dispatching policy: GPU is the first (primary) server; CPU is backup.
/// Jobs with estimated GPU execution time < τ go to the GPU, otherwise CPU.
/// Every m invocations, adjust τ by ±epsilon to drive observed GPU load toward
/// ρ̃₁ = ρ + α·ρ⁴·(1−ρ).
pub struct Mice {
    que_map: QueueMap,
    cmap: WorkerCharMap,
    state: Mutex<MiceState>,
    /// Epoch length (jobs per threshold update).
    m: u64,
    /// Threshold step size (seconds).
    epsilon: f64,
    /// α parameter in the target load formula.
    alpha: f64,
    clock: Clock,
}

impl Mice {
    fn get_gpu_est(&self, fqdn: &str, mqfq_est: f64) -> (f64, f64) {
        use iluvatar_library::char_map::Value;

        // Prior filtered estimate and prior observed E2E time.
        let (prev_est_raw, prev_e2e_raw) =
            self.cmap
                .get_2(fqdn, Chars::EstGpu, Value::Avg, Chars::E2EGpu, Value::Avg);

        // Clamp/repair any missing or negative observations to avoid learning sentinels.
        let prev_est = if prev_est_raw.is_finite() && prev_est_raw > 0.0 {
            prev_est_raw
        } else {
            mqfq_est.max(0.0)
        };
        let prev_e2e = if prev_e2e_raw.is_finite() && prev_e2e_raw > 0.0 {
            prev_e2e_raw
        } else {
            mqfq_est.max(0.0)
        };
        let obs = if mqfq_est.is_finite() && mqfq_est > 0.0 {
            mqfq_est
        } else {
            prev_est
        };

        // Same “Kalman-like” blending used in landlord.
        let z = prev_e2e - prev_est; // residual
        let alpha = 0.1;
        let beta = 0.7;
        let k = 1.0 - (beta + alpha); // = 0.2
        let xhat = (alpha * prev_est) + (beta * obs) + k * z;

        self.cmap.update(fqdn, Chars::EstGpu, xhat);

        // Log estimator inputs/outputs for offline analysis.
        info!(
            fqdn = fqdn,
            raw_est = mqfq_est,
            obs = obs,
            prev_est = prev_est,
            prev_e2e = prev_e2e,
            residual = z,
            xhat = xhat,
            alpha = alpha,
            beta = beta,
            k = k,
            "MICE_EST: gpu_estimator_update"
        );

        (xhat, z)
    }

    /// Construct a new Mice policy (parameters can be wired from config later).
    pub fn new(
        _invocation_config: Arc<InvocationConfig>,
        cmap: WorkerCharMap,
        que_map: QueueMap,
        tid: &TransactionId,
    ) -> Result<Self> {
        let clock = get_global_clock(tid)?;
        Ok(Self {
            que_map,
            cmap,
            state: Mutex::new(MiceState {
                tau: 10.0, // conservative initial τ (seconds)
                gpu_work: 0.0,
                cpu_work: 0.0,
                count: 0,
                last_update: clock.now(),
            }),
            m: 100,       // epoch length (jobs)
            epsilon: 0.1, // τ step (seconds)
            alpha: 0.8,   // target-load α
            clock,
        })
    }

    /// Use (filtered) GPU exec time as the job "size"; fall back to char map.
    fn job_size_est(&self, fid: &str, filtered_gpu_est: f64) -> f64 {
        if filtered_gpu_est.is_finite() && filtered_gpu_est > 0.0 {
            return filtered_gpu_est;
        }
        let est = self.cmap.get_avg(fid, Chars::GpuExecTime);
        if est.is_finite() && est > 0.0 {
            est
        } else {
            1.0 // conservative fallback
        }
    }
}

impl DispatchPolicy for Mice {
    fn choose(&self, reg: &Arc<RegisteredFunction>, tid: &TransactionId) -> (Compute, f64, f64) {
        // Pull queue estimates for GPU/CPU.
        let (gpu_est, gpu_load) = match self.que_map.get(&Compute::GPU) {
            Some(q) => q.est_completion_time(reg, tid),
            None => {
                warn!(tid = %tid, fqdn = %reg.fqdn, "MICE_WARN: gpu_queue_missing");
                (NO_ESTIMATE, NO_ESTIMATE)
            }
        };
        let (cpu_est, cpu_load) = match self.que_map.get(&Compute::CPU) {
            Some(q) => q.est_completion_time(reg, tid),
            None => {
                warn!(tid = %tid, fqdn = %reg.fqdn, "MICE_WARN: cpu_queue_missing");
                (NO_ESTIMATE, NO_ESTIMATE)
            }
        };

        // Smooth GPU execution-time estimate (local copy of landlord's function).
        let (gpu_est_exec, _err) = self.get_gpu_est(&reg.fqdn, gpu_est);
        let size = self.job_size_est(&reg.fqdn, gpu_est_exec);

        // GPU is ALWAYS the first server in the sequential order for this policy.
        // Route to GPU iff size < τ, else to CPU. If no GPU queue, fallback to CPU.
        let now = self.clock.now();
        let mut st = self.state.lock();
        let gpu_available = self.que_map.get(&Compute::GPU).is_some();
        let use_gpu = gpu_available && (size < st.tau);

        // Pre-decision logging (inputs).
        info!(
            tid = %tid,
            fqdn = %reg.fqdn,
            tau = st.tau,
            job_size = size,
            gpu_available = gpu_available,
            est_gpu_completion = gpu_est,
            est_cpu_completion = cpu_est,
            gpu_load = gpu_load,
            cpu_load = cpu_load,
            "MICE_DECIDE: inputs"
        );

        // Update per-epoch workload counters using the job-size proxy.
        if use_gpu {
            st.gpu_work += size;
        } else {
            st.cpu_work += size;
        }
        st.count += 1;

        // End of epoch? adjust τ using the target-load rule: ρ̃₁ = ρ + α·ρ⁴·(1−ρ)
        if st.count >= self.m && st.last_update < now {
            let dt = (now - st.last_update).as_seconds_f64().max(1e-6);
            let rho_gpu = st.gpu_work / dt;
            let rho_cpu = st.cpu_work / dt;
            let rho = rho_gpu + rho_cpu;

            let target_gpu = rho + self.alpha * (rho.powi(4)) * (1.0 - rho);

            let tau_old = st.tau;
            if rho_gpu < target_gpu {
                st.tau += self.epsilon;
            } else {
                st.tau = (st.tau - self.epsilon).max(0.0);
            }

            // Epoch update logging (outputs).
            info!(
                tid = %tid,
                epoch_m = self.m,
                dt = dt,
                rho_total = rho,
                rho_gpu = rho_gpu,
                rho_cpu = rho_cpu,
                alpha = self.alpha,
                epsilon = self.epsilon,
                target_gpu = target_gpu,
                tau_old = tau_old,
                tau_new = st.tau,
                gpu_work = st.gpu_work,
                cpu_work = st.cpu_work,
                "MICE_EPOCH: threshold_update"
            );

            st.gpu_work = 0.0;
            st.cpu_work = 0.0;
            st.count = 0;
            st.last_update = now;
        }
        drop(st);

        // Return device, load, and est. completion time for enqueue path.
        let (dev, load, est) = if use_gpu {
            (Compute::GPU, gpu_load, gpu_est)
        } else {
            (Compute::CPU, cpu_load, cpu_est)
        };

        // Post-decision logging (result).
        info!(
            tid = %tid,
            fqdn = %reg.fqdn,
            device = ?dev,
            job_size = size,
            tau_current = self.state.lock().tau,
            chosen_load = load,
            chosen_est_completion = est,
            "MICE_DECIDE: output"
        );

        (dev, load, est)
    }
}
