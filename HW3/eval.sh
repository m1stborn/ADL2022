#python eval.py -r ./assets/public.jsonl -s ./assets/result-origin.jsonl
#python eval.py -r ./assets/public.jsonl -s ./assets/result-beams_2.jsonl
#python eval.py -r ./assets/public.jsonl -s ./assets/result-beams_4.jsonl
#python eval.py -r ./assets/public.jsonl -s ./assets/result-temp_0.7.jsonl
#python eval.py -r ./assets/public.jsonl -s ./assets/result-top_50.jsonl
python eval.py -r ./assets/public.jsonl -s ./assets/result-temp_0.9.jsonl
python eval.py -r ./assets/public.jsonl -s ./assets/result-top_30.jsonl
