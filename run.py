
import os
import sys

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

transformers = environmentinator.ensure_module('transformers')

codetiming = environmentinator.ensure_module('codetiming')

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


def main(args=sys.argv):
    with codetiming.Timer(text='{:.2f}s: Entire program run time (incl. model load)'):
        with codetiming.Timer(text='{:.2f}s: Model load time'):
            model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        with codetiming.Timer(text='{:.2f}s: Question-answering time'):
            question = '''What color is car 3?'''

            questions_and_answer = [
                ('What color is car 1?', 'blue'),
                ('What kind of car is car 1?', 'sedan'),
                ('What color is car 2?', 'white'),
                ('What kind of car is car 2?', 'Jeep'),
                ('What color is car 3?', 'black'),
                ('What color is car 4?', 'green'),
            ]

            paragraph = ''' There is a park with five cars parked. The first car is a blue sedan. The second car is a Jeep colored white. The third car is 
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


