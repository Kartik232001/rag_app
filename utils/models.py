import ollama
import subprocess

def fetch_models():
    lists = ollama.list()
    models = []
    for model in lists['models']:
        if 'embed' not in model['name'].lower():
            models.append(model['name'])
        else:
            embed_model_install()
    return models

def embed_model_install():
    try:
        result = subprocess.run(["ollama", "pull", "nomic-embed-text"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            print("Embedding model installed")
            print("Output:\n", result.stdout)
        else:
            print("Embedding model install failed with error:")
            print(result.stderr)

    except Exception as e:
        print(f"Failed to run 'ollama run': {e}")
