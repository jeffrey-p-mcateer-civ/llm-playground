
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


from transformers import BertForQuestionAnswering
from transformers import BertTokenizer


def main(args=sys.argv):
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    question = '''What is Machine Learning?'''

    paragraph = ''' Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance 
                    on a specific task. Machine learning algorithms build a mathematical model of sample data, known as "training data", in order to make predictions or 
                    decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in the applications of email filtering, detection 
                    of network intruders, and computer vision, where it is infeasible to develop an algorithm of specific instructions for performing the task. Machine learning 
                    is closely related to computational statistics, which focuses on making predictions using computers. The study of mathematical optimization delivers methods, 
                    theory and application domains to the field of machine learning. Data mining is a field of study within machine learning, and focuses on exploratory 
                    data analysis through unsupervised learning.In its application across business problems, machine learning is also referred to as predictive analytics. '''
                
    encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)

    inputs = encoding['input_ids']  #Token embeddings
    sentence_embedding = encoding['token_type_ids']  #Segment embeddings
    tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

    #start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
    start_scores, end_scores = model(torch.tensor([inputs]),token_type_ids=torch.tensor([sentence_embedding]), return_dict=False)

    print(f'start_scores = {start_scores}  end_scores = {end_scores}')

    start_index = torch.argmax(start_scores)

    end_index = torch.argmax(end_scores)

    answer = ' '.join(tokens[start_index:end_index+1])

    corrected_answer = ''

    for word in answer.split():
        
        #If it's a subword token
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word

    print()
    print(f'paragraph = {paragraph}')
    print()
    print(f'question = {question}')
    print()
    print(f'corrected_answer = {corrected_answer}')
    print()
    




if __name__ == '__main__':
    main()


