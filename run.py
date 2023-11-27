
import os
import sys
import shutil
import subprocess
import traceback

if not 'TRANSFORMERS_CACHE' in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'llm-models'
    )

print(f'Using TRANSFORMERS_CACHE = {os.environ["TRANSFORMERS_CACHE"]}')
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

import environmentinator

#torch = environmentinator.ensure_module('torch')
# CUDA Acceleration
#torch = environmentinator.ensure_module('torch', 'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
# Plain CPU processing
torch = environmentinator.ensure_module('torch', 'torch torchvision torchaudio')

# TODO possibly see cuda repo urls: https://huggingface.co/TheBloke/Yi-34B-GPTQ#how-to-use-this-gptq-model-from-python-code
transformers = environmentinator.ensure_module('transformers', 'transformers optimum auto-gptq')

codetiming = environmentinator.ensure_module('codetiming')

#from transformers import BertForQuestionAnswering
#from transformers import BertTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Grab a copy of https://github.com/lowleveldesign/process-governor#limit-memory-of-a-process
# and attempt to limit our processes RAM working set to something like 12gb
if os.name == 'nt':
    import urllib.request
    import zipfile
    if not shutil.which('procgov64'):
        dl_url = 'https://github.com/lowleveldesign/process-governor/releases/download/2.12/procgov.zip'
        dl_zipfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'llm-models', 'procgov.zip')
        if not os.path.exists(dl_zipfile):
            urllib.request.urlretrieve(dl_url, dl_zipfile)
        dl_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'llm-models', 'procgov')
        os.makedirs(dl_folder, exist_ok=True)
        os.environ['PATH'] = dl_folder + ';' + os.environ['PATH']
        if not shutil.which('procgov64'):
            with zipfile.ZipFile(dl_zipfile, 'r') as zip_ref:
                zip_ref.extractall(dl_folder)




def main(args=sys.argv):
    if os.name == 'nt' and shutil.which('procgov64'):
        try:
            subprocess.run([
                'procgov64', '--minws', '1200M', '--maxws', '12000M', '-p', f'{os.getpid()}'
            ])
        except:
            traceback.print_exc()
    with codetiming.Timer(text='{:.2f}s: Entire program run time (incl. model load)'):
        with codetiming.Timer(text='{:.2f}s: Model load time'):
            #model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            #model_path = "TheBloke/Yi-34B-GPTQ"
            model_path = "TheBloke/Yi-34B-GGUF"
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                             device_map="auto",
                                             # trust_remote_code=True,
                                             revision="main")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)


        with codetiming.Timer(text='{:.2f}s: Question-answering time'):
            
            questions_and_answer = [
                ('How many cars are parked?', '4'),
                ('What color is car 1?', 'blue'),
                ('What kind of car is car 1?', 'sedan'),
                ('What color is car 2?', 'white'),
                ('What kind of car is car 2?', 'Jeep'),
                ('What color is car 3?', 'black'),
                ('What color is car 4?', 'green'),
                ('What color is car 5?', 'no data!'),
            ]

            paragraph = ''' There is a park with cars parked. The first car is a blue sedan. The second car is a Jeep colored white. The third car is 
                                is a black van with baloons attached to the top. Car number four is a green sedan. There are trees surrounding the cars.
                '''
            
            print()
            print(f'paragraph = {paragraph}')
                
            for question, expected_answer in questions_and_answer:
                            
                encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)

                inputs = encoding['input_ids']  #Token embeddings
                sentence_embedding = encoding['token_type_ids']  #Segment embeddings
                tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

                #start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
                start_scores, end_scores = model(torch.tensor([inputs]),token_type_ids=torch.tensor([sentence_embedding]), return_dict=False)

                start_index = torch.argmax(start_scores)

                end_index = torch.argmax(end_scores)

                answer = ' '.join(tokens[start_index:end_index+1])

                llm_answer = ''

                for word in answer.split():
                    
                    #If it's a subword token
                    if word[0:2] == '##':
                        llm_answer += word[2:]
                    else:
                        llm_answer += ' ' + word

                print()
                print(f'question = {question}')
                print(f'llm_answer = {llm_answer}')
                print(f'expected_answer = {expected_answer}')
                print()
        




if __name__ == '__main__':
    main()


