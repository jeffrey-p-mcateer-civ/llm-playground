

import os
import json

llm_model_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'llm-models'
)
if r'\\' in llm_model_path:
    llm_model_path = 'S:\\'+llm_model_path[llm_model_path.index('Users'):]
os.makedirs(llm_model_path, exist_ok=True)
print(f'llm_model_path = {llm_model_path}')
if not 'TRANSFORMERS_CACHE' in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = llm_model_path

import environmentinator

gpt4all = environmentinator.ensure_module('gpt4all')
pprint = environmentinator.ensure_module('pprint')
codetiming = environmentinator.ensure_module('codetiming')

# See https://gpt4all.io/index.html#Model%20Explorer
# https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf

with codetiming.Timer('Load Model'):
    model = gpt4all.GPT4All(
        model_name="mistral-7b-instruct-v0.1.Q4_0.gguf",
        model_path=llm_model_path,
        allow_download=True
    )

# with model.chat_session():
    
#     # with codetiming.Timer('Generate Output'):
#     #     response = model.generate("Give me a list of 10 colors and their RGB code")
#     # print(response)
#     # print()
#     # pprint.pprint(model.current_chat_session)
#     # print()

#     scene_description = '''
# It is a bright sunny day outside.
# I have a park with four dogs in it. The first dog is by the road, and is named Sam.
# The second dog, named John, is in the pool. Spot is under a tree, and Kevin is barking at a car.
# There is a newspaper near the car that nobody is reading.
# '''.strip()

#     tq1 = f'''
# {scene_description}

# Output the closest thing to John in machine JSON format.
# '''.strip()

#     print(f'prompt> {tq1}')
#     with codetiming.Timer('Generate Output'):
#         response = model.generate(tq1)

#     print(response)

# with model.chat_session():

#     tq2 = f'''
# Remove all text except machine-parseable JSON from the following document:

# {response}
# '''.strip()
    
#     print('=' * 12, 'JSON only', '=' * 12)
#     with codetiming.Timer('Generate Output'):
#         response = model.generate(tq2)

#     print(response)

# with model.chat_session():

#     tq3 = f'''
# {scene_description}

# Output all data above as machine-parseable JSON
# '''.strip()

#     print('=' * 12, 'More JSON experiments', '=' * 12)
#     with codetiming.Timer('Generate Output'):
#         response = model.generate(tq2)

#     print(response)

# New session for interactive mode

def maybe_re_prompt_for_json(model, original_response, remaining_recursions=2):
    try:
        json.loads(original_response)
        return original_response
    except:
        simple_transform_begin = original_response.find('{')
        simple_transform_end = original_response.rfind('}')
        if simple_transform_begin >= 0 and simple_transform_begin < len(original_response) and simple_transform_end >= 0 and simple_transform_end < len(original_response):
            try:
                simple_transform = original_response[simple_transform_begin:simple_transform_end+1].strip()
                json.loads(simple_transform)
                return simple_transform
            except:
                pass
        
        if remaining_recursions < 1:
            print()
            print(original_response)
            print()
            raise
        print('** re-prompting for JSON **')
        new_prompt = f'''
{original_response}

Output all data above as a single machine-parseable JSON dictionary.
'''.strip()
        return maybe_re_prompt_for_json(model, model.generate(new_prompt), remaining_recursions-1)


system_template = '''
A chat between an analyst and a machine which outputs responses as machine-parseable JSON
'''.strip()
print('=' * 24)
print(system_template)
print('=' * 24)
with model.chat_session(system_template):

    while True:
        print()
        question = input('prompt> ').strip()
        if question == 'q' or question == 'e':
            break
        if len(question) > 1:
            
            if '\\n' in question:
                question = question.replace('\\n', '\n')

            with codetiming.Timer('Generate Output'):
                response = maybe_re_prompt_for_json( model, model.generate(question) )
            print(response)
            #pprint.pprint(model.current_chat_session)






