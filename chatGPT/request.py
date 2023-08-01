from utils.file import read_file, write_file
import openai
import time
from retrying import retry
from utils.log import getLogger
import logging
from tqdm import tqdm
import random
import os

logger = getLogger(args="INFO", name="request")

@retry(wait_random_min=2000, wait_random_max=5000)
def send_request(
    prompt, 
    model="gpt-3.5-turbo",
    temperature=0.01, 
    max_tokens=2048,
    top_p=1.0,
    n=1,
    frequency_penalty=0.0,
):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Recover the text represented by the Abstract Meaning Representation graph (AMR graph) enclosed within triple quotes. Utilize only the information provided in the input. Output only the recovered text."
            },
            {
                "role": "user",
                "content": '''"""{prompt}"""'''.format(prompt=prompt)
            }
        ],
        temperature=temperature,
        top_p=top_p,
        n=n,
        frequency_penalty=frequency_penalty
    )
    response = response["choices"][0]["message"]["content"]
    time.sleep(0.2)
    return str(response)
            
def request(args):
    logger.info(args)
    openai.organization = args.organization
    openai.api_key = args.API_key
    logger.info("start reading from file {file}".format(file=args.data_path))
    data = read_file(args.data_path)
    data = ''.join(data)
    input = data.split('\n\n')
    logger.info("finished reading from file {file}".format(file=args.data_path))
    logger.info("start reading from file {file}".format(file=args.target_path))
    target = read_file(args.target_path)
    data = []
    for item_input, item_target in zip(input, target):
        data.append({
            "input": item_input,
            "target": item_target
        })
    data = random.sample(data, 50)
    logger.info("finished reading from file {file}".format(file=args.target_path))
    # random select data from data and target with the same index
    logging.disable(logging.CRITICAL)
    result = []
    target = []
    for item in tqdm(data):
        result.append(send_request(item["input"]).replace('\n', ''))
        target.append(item["target"][:-1])
    logging.disable(logging.NOTSET)
    logger.info("start writing to file {file}".format(file=os.path.join(args.output_path, "result.txt")))
    write_file(os.path.join(args.output_path, "result.txt"), result)
    logger.info("finished writing to file {file}".format(file=os.path.join(args.output_path, "result.txt")))
    logger.info("start writing to file {file}".format(file=os.path.join(args.output_path, ".txt")))
    write_file(os.path.join(args.output_path, "target.txt"), target)
    logger.info("finished writing to file {file}".format(file=os.path.join(args.output_path, "target.txt")))
