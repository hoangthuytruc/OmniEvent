import json 
import unittest
import sys 
sys.path.append("..")
from OmniEvent.infer import infer
from tqdm import tqdm

class TestInfer(unittest.TestCase):

    def test_seq2seq(self):
        input_text = "U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn"
        result = infer(task="EE", text=input_text)
        result = sorted(result[0]["events"], key=lambda item: item["trigger"])
        print(json.dumps(result, indent=4))
        self.assertEqual(result[0]["trigger"], "assault")
        self.assertEqual(result[0]["type"], "attack")
        self.assertEqual(result[1]["trigger"], "pounded")
        self.assertEqual(result[1]["type"], "injure")

def write_results(data: list, io_path: str):
    with open(io_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    eval_path = 'path/to/file/eval.json'
    output_path = './tests/eval-output.json'

    eval_data = []
    with open(eval_path) as f:
        eval_data = json.load(f)
    
    ans = []
    try:
        for instance in tqdm(eval_data):
            id = instance["id"]
            text = instance["text"]
            result = infer(task="EE", text=text)
            events = result[0]["events"]
            ans.append({
                "id": id,
                "text": text,
                "events": events
            })
    except:
        write_results(ans, output_path)
    
    write_results(ans, output_path)