import json, glob, os

def main():
    path = '/data/learn-ai/rag-test/results/'
    files = glob.glob(os.path.join(path, '*_both.json'))
    deleted_count = 0
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            
            if data.get('agent_llm_model') == 'gemma':
                os.remove(f)
                print(f"Deleted {os.path.basename(f)}")
                deleted_count += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print(f"Total files deleted: {deleted_count}")

if __name__ == "__main__":
    main()
