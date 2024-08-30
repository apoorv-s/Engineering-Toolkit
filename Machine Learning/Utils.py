# Commonly used functions

def generate_description(config, run_number = None):
    config_vars = vars(config)
    config_str = f"Run number: {run_number}\nConfig:\n"
    config_str += "\n".join(f"    {key} = {repr(value)}" for key, value in config_vars.items())
    return config_str