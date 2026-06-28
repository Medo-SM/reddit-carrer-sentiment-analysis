#### Usage Demo
```
import yaml

# 1. Load the configuration file
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Access variables just like a standard Python dictionary
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
should_lowercase = config["preprocessing"]["lowercase"]
```
