import transformers
import torch


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

initial_msg = {"role": "system", "content": "You are a chatbot that answers users questions. If user gives a context, answer based on the context only, if you can't find info in the context, say i don't know."}
messages = [initial_msg]
  
def chat(question, context=''):
    if len(context)>3:
        messages.append({"role": "user", "content": f'###questions:{question} ###context:{context}'})
        outputs = pipeline(
            messages,
            max_new_tokens=800,
        )
    else:
        messages.append({"role": "user", "content": question})
        outputs = pipeline(
            messages,
            max_new_tokens=800,
        )
    
    messages.append(outputs[0]['generated_text'][-1])
    return outputs[0]['generated_text'][-1]['content']
    

def clear_chat_history():
    messages = [initial_msg]
    return messages
    



chat("what is my last question? I forgot it")
