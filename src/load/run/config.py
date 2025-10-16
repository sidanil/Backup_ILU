from typing import Dict
from enum import Enum

def _primitive(v):
    if isinstance(v, Enum):
        return v.value
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple, set)):
        return [_primitive(x) for x in v]
    if isinstance(v, dict):
        return {k: _primitive(x) for k, x in v.items()}
    # last resort
    return str(v)



class ConfigItem:
    def __init__(self, name, category, default, tree=None):
        self.name = name
        self.category = category
        self.default = default
        self.tree = tree

    def formattable(self) -> bool:
        return self.tree != None

    def update_json(self, json_config):
        c = json_config
        if self.tree == None:
            print(self.name)
        for p in self.tree[:-1]:
            if p in c:
                pass
            else:
                c[p] = dict()
            c = c[p]
        c[self.tree[-1]] = self.default

    def to_env_var(self, prepend: str) -> str:
        ret = prepend
        for t in self.tree:
            ret += "__" + t
        return ret

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "default": _primitive(self.default),
            "tree": list(self.tree) if self.tree is not None else None,
        }


class LoadConfig:
    def __init__(self):
        self.storage = {}
        self.env_vars = {}

    def __getitem__(self, key):
        return self.storage[key].default

    def __setitem__(self, key, value):
        if key not in self:
            new_item = ConfigItem(key, "UNKNOWN", value, None)
            self.storage[key] = new_item
        self.storage[key].default = value

    def __contains__(self, key):
        return key in self.storage

    def insert(self, item: ConfigItem):
        self.storage[item.name] = item

    def bulk_add(self, category, items, env_var = None):
        if env_var is not None:
            self.env_vars[category] = env_var
        for item in items:
            if len(item) == 2:
                name, default = item
                config_item = ConfigItem(name, category, default, None)
            if len(item) == 3:
                name, default, tree = item
                config_item = ConfigItem(name, category, default, tree)
            self.insert(config_item)

    def overwrite(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v
        return self

    def to_json(self, category, json_data):
        configs = filter(lambda x: x.category == category, self.storage.values())
        configs = filter(lambda x: x.default != None, configs)
        for item in configs:
            item.update_json(json_data)

    def to_env_var_dict(self, category) -> Dict[str, str]:
        ret = {}
        configs = filter(lambda x: x.category == category, self.storage.values())
        configs = filter(lambda x: x.default is not None, configs)
        configs = filter(lambda x: x.tree is not None, configs)
        for item in configs:
            varname = item.to_env_var(self.env_vars[category])
            ret[varname] = str(item.default)
        return ret
    
    def to_dict(self) -> dict:
        # Flat: name -> value
        flat = {name: _primitive(item.default) for name, item in self.storage.items()}

        # By category: category -> {name -> value}
        by_category = {}
        for item in self.storage.values():
            by_category.setdefault(item.category, {})[item.name] = _primitive(item.default)

        # Structured per category using `tree` (same placement logic as update_json)
        structured = {}
        categories = {item.category for item in self.storage.values()}
        for cat in categories:
            data = {}
            for item in self.storage.values():
                if item.category != cat or item.tree is None or item.default is None:
                    continue
                c = data
                for p in item.tree[:-1]:
                    c = c.setdefault(p, {})
                c[item.tree[-1]] = _primitive(item.default)
            structured[cat] = data

        return {
            "flat": flat,
            "by_category": by_category,
            "structured": structured,
            "env_var_prefixes": dict(self.env_vars),
        }
