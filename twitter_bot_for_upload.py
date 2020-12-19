import json
from googlesearch import search
import tweepy
from simpletransformers.question_answering import QuestionAnsweringModel
from bs4 import BeautifulSoup
import requests
import html2text

def intitilize_twitter_bot():
    #Set the tokens to my personal twitter developer information

    ##CODE TO ADD##
    #You must add your own twitter developer codes in the example code provided below
    #ACCESS_TOKEN = ''
    #ACCESS_TOKEN_SECRET = ''
    #API_KEY = ''
    #API_SECRET = ''

    #Initialize the tokens into variables
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    #Use the methods imported from Tweepy in order to gain access to Twitter
    api = tweepy.API(auth)
    return api

def get_question_from_mentions(mention):
    user_data = []
    user_question = mention.text
    #parse the data so that we only have the question
    split_question = user_question.split(' ', 1)
    parsed_question = split_question[1]
    user_id = split_question[0]
    #print('\n' + parsed_question)
    #Ask a question using google search
    question = str(parsed_question)
    user_data.append(user_id)
    user_data.append(question)
    return user_data

def get_context(question):
    question_context = []
    #Tried concatinating 'Wikipedia' on each question to get factual information to
    #show up in the first three links
    question = str(question)
    #This for loop uses the google-search library to search our question on google.
    #It then returns the first three links which we add to our question contexts
    #to later be formatted
    for query in search(question, tld="com", lang='en', stop=5, pause=2.0):
        formatted_text = format_google_query(query)
        question_context.append(formatted_text)
    return question_context

def format_google_query(query):
    #Sets up an html2Text object that can be later used to parse the html for us
    text_parser = html2text.HTML2Text()
    #The lines below are focused on making the html easier for bert to read and gather
    #context from. If we remove things we dont need we have a higher likelihood of stumbling
    #upon the right information
    text_parser.ignore_links = True
    text_parser.ignore_anchors = True
    text_parser.ignore_images = True
    text_parser.escape_snob = True
    text_parser.ignore_emphasis = True
    #Uses the requests library to access the html at the link provided by 'query'
    request = requests.get(query)
    #Uses the html parser to go through the text obtained through the request library
    text = text_parser.handle(request.text)
    return text

#Use the Bert model to make an educated guess with the question and context inputted
def predict_answer(model, question, contexts, max_len=512):
    split_context = []

    #This checks if our context is a list and if not, it converts it to a list
    #Realistically we should always have three links so we shouldnt ever have to worry
    #about just one list item
    if not isinstance(contexts, list):
        contexts = [contexts]

    #breaks the contextual data down into smaller chunks
    for context in contexts:
        for i in range(0, len(context), max_len):
            split_context.append(context[i:i+max_len])

    split_context = contexts

    f_data = []

    #The format that the bert model requires in order to read the question and process
    #the contexual data
    for i, c in enumerate(split_context):
        f_data.append(
            {'qas':
              [{'question': question,
               'id': i,
               'answers': [{'text': ' ', 'answer_start': 0}],
               'is_impossible': False}],
              'context': c
            })

    prediction = model.predict(f_data)
    return prediction

#A function to break down the prediction that comes as a dictionary of lists. This function
#finds the highest probability and then finds the answer that correlates to the id of then
#highest probability. It then prints both the probability and answer.
def evaluate_prediction(user, question, prediction):
    probability = 0
    #first for loop is to find the highest probability and save the id_value
    for index in range (len(prediction)):
        for key in range (len(prediction[index])):
            string = prediction[index][key]
            #Takes out the empty list
            if string == prediction[index][key]:
                #first case to initialize probability to something
                if string.get('probability'):
                    values = string.get('probability')
                    if values[0] > probability:
                        probability = values[0]
                        id_value = string.get('id')

    #second for loop to find the answer that matches the id_value
    for index in range (len(prediction)):
        string = prediction[index][id_value]
        if string.get('answer'):
            values = string.get('answer')
            answer = values[0]

    return ('\n\n' 'You asked: ' + question + '\nAnswer: ' + answer + ' with a ' + str(round(probability * 100, 2)) + '% probability\n')

#Short function to reply with the answer on twitter
def reply_with_answer(api, user_id, evaluation):
    api.update_status(evaluation, in_reply_to_status_id = user_id, auto_populate_reply_metadata = True)

def main():
    print('\nRunning...')
    #initialize the twitter bot with your personal twitter developer keys
    api = intitilize_twitter_bot()
    #get the mentions from your twitter timeline
    mentions = api.mentions_timeline()
    #a boolean to check if the first mention is being looked at. This will be changed
    #in the future when I update the bot to look for tweets in real time.
    first_mention = True
    for mention in mentions:
        if first_mention == True:
            #Finds the question from the current mention being looked at
            user_data = get_question_from_mentions(mention)
            user_id = str(mention.id)
            question = str(user_data[1])
            question_context = get_context(question)

            #if you would like to train a model on a different data-set, use the statement below
            #model = QuestionAnsweringModel('bert', 'bert-base-cased', use_cuda=False, args=train_args)

            #This statement initializes the model with the distilbert dataset and makes sure that we are not using
            #CUDA on our machine. CUDA is a feature that is used with NVIDIA graphics cards in order to compute
            #and execute the model at a faster speed. If you do not have a NVIDIA machine or CUDA installed, leave
            #use_cuda equal to False.
            model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', use_cuda=False)

            #Get the prediction from the model
            prediction = predict_answer(model, question, question_context)
            #Evaulate the information in the prediction datatype
            evaluation = evaluate_prediction(user_id, question, prediction)
            print(evaluation)
            reply_with_answer(api, user_id, evaluation)
            first_mention = False
            
    print('\n\nExiting...')

if __name__ == "__main__":
    main()
