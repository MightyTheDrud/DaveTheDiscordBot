import discord
from discord.ext import commands
from datetime import datetime, timedelta
from dateutil import relativedelta
from pytz import timezone
import os
from openai import OpenAI
import random
import requests
import re
import jsons
import json
import tempfile
import unicodedata
import base64
import numpy as np
from bs4 import BeautifulSoup
import scrapetube
import sys
from collections import Counter
import math
import yfinance as fn
from PIL import Image as PilImage
from PIL.ExifTags import TAGS, GPSTAGS
from forex_python.converter import CurrencyRates
from forex_python.bitcoin import BtcConverter
import time
import asyncio
import json  
import hashlib  
import pickle  
from sentence_transformers import SentenceTransformer, util 
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain.chains import LLMChain
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
import NewGPTKey
import CarterTrivia


"""
Dave the Discord Bot - A comprehensive array of functionalities are covered below, from eBay pricing to chatGPT functionality. Certain tweaks have been made to maximize functionality 
on low-powered devices, such as the Raspberry Pi or an old laptop collecting dust in your closet.
"""


TxtDocReady = asyncio.Event()

os.environ["OPENAI_API_KEY"] = NewGPTKey.GPTKEY
ChatGPTKey = NewGPTKey.GPTKEY
client = OpenAI(api_key = ChatGPTKey)

intents = discord.Intents.all()
intents.members = True  # Enables the member update event
bot = commands.Bot(command_prefix='!', intents=intents)

# Precomputed tag embeddings for PlutoPhotos
tag_embeddings_cache = {}
photo_tags_cache = {}


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print('------')
    
    # Precompute tag embeddings for PlutoPhotos at startup
    global tag_embeddings_cache, photo_tags_cache, plutoImages, tag_weights, embeddings_cache_path, metadata_cache_path
    home_dir = os.path.expanduser("~")
    plutoPhotoPath = os.path.join(home_dir, "PlutoPhotos")
    jsonPath = os.path.join(plutoPhotoPath, "photoTags.json")
    embeddings_cache_path = os.path.join(plutoPhotoPath, "tag_embeddings_cache.pkl")
    metadata_cache_path = os.path.join(plutoPhotoPath, "tag_metadata_cache.pkl")

    # Load images from PlutoPhotos folder
    plutoImages = [file for file in os.listdir(plutoPhotoPath) if file.endswith(("jpg", "jpeg", "png", "mpo"))]

    if not plutoImages:
        print("No images found in the PlutoPhotos folder... ")
        return

    # Load tags from photoTags.json
    if not os.path.exists(jsonPath):
        print("The photo tags file (photoTags.json) is missing.")
        return

    with open(jsonPath, "r") as f:
        photo_tags_cache = json.load(f)

    # Check if there are any tagged photos
    if not photo_tags_cache:
        print("No tagged photos found in photoTags.json.")
        return
    
    # Compute tag weights
    tag_weights = compute_tag_weights(photo_tags_cache)
    print(f"Computed weights for {len(tag_weights)} unique tags.")

    # Precompute embeddings in a separate thread, loading from cache if available
    tag_embeddings_cache = await asyncio.to_thread(precompute_embeddings, plutoImages, photo_tags_cache)
    
    global TxtDoc, SplitInput, AllSplitInput, vectorstore, retriever
    TxtDoc = []  
    
    try:
        path = "./TrainingData/"
        text_loader_kwargs={'autodetect_encoding': True}
        loader = DirectoryLoader(path, glob="**/*.txt", loader_cls = TextLoader, loader_kwargs = text_loader_kwargs)

        TxtDoc = loader.load()	

        SplitInput = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100, add_start_index = True)
    
        AllSplitInput = SplitInput.split_documents(TxtDoc)
        
        virtualDaveVectorstorePath = "/mnt/usb/VirtualDaveVectorstore/"
        
        if os.path.exists(virtualDaveVectorstorePath):
            vectorstore = Chroma(persist_directory = virtualDaveVectorstorePath, embedding_function = OpenAIEmbeddings())
        
        else:
            vectorstore = Chroma.from_documents(documents = AllSplitInput, embedding = OpenAIEmbeddings(), persist_directory = virtualDaveVectorstorePath)
            vectorstore.persist()
    
        retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})
        
        TxtDocReady.set()
        
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Something wrong happened in if statement. Try again...")
    
    
def compute_tag_hash(tags):
    """Compute a hash of the tags to detect changes."""
    tag_string = " ".join(sorted(tags))  # Sort tags to ensure consistent hashing
    return hashlib.md5(tag_string.encode('utf-8')).hexdigest()


def precompute_embeddings(plutoImages, photo_tags):
    """Precompute embeddings for new or updated photos and update the cache."""
    print("Starting precomputation of tag embeddings...")
    start_time = time.time()
    embedder = SentenceTransformer('all-MiniLM-L12-v2')

    # Load existing cache and metadata if they exist
    cached_embeddings = {}
    cached_metadata = {}
    if os.path.exists(embeddings_cache_path) and os.path.exists(metadata_cache_path):
        with open(embeddings_cache_path, 'rb') as f:
            cached_embeddings = pickle.load(f)
        with open(metadata_cache_path, 'rb') as f:
            cached_metadata = pickle.load(f)
        print(f"Loaded cached embeddings for {len(cached_embeddings)} photos.")

    # Determine which photos need new embeddings (new photos or updated tags)
    photos_to_process = []
    new_metadata = {}
    for idx, (filename, data) in enumerate(photo_tags.items()):
        if filename not in plutoImages:  # Skip if the photo no longer exists
            continue
        tags = data.get("tags", [])
        tag_hash = compute_tag_hash(tags)

        # Check if the photo is new or its tags have changed
        if (filename not in cached_embeddings or
                filename not in cached_metadata or
                cached_metadata[filename]['tag_hash'] != tag_hash):
            photos_to_process.append((filename, tags))
        else:
            # Use the cached embedding if the tags haven't changed
            tag_embeddings_cache[filename] = cached_embeddings[filename]

        # Update metadata
        new_metadata[filename] = {'tag_hash': tag_hash}

        # Log progress every 50 photos to monitor performance
        if (idx + 1) % 50 == 0:
            elapsed_time = time.time() - start_time
            print(f"Checked {idx + 1}/{len(photo_tags)} photos in {elapsed_time:.2f} seconds.")

    # Precompute embeddings for new or updated photos
    if photos_to_process:
        print(f"Computing embeddings for {len(photos_to_process)} new or updated photos...")
        for idx, (filename, tags) in enumerate(photos_to_process):
            # Join tags into a single string for embedding, giving more weight to location tags
            weightedTags = tags + [tag for tag in tags if tag in [tags[-2], tags[-1]]] * 3
            tagSentence = " ".join(weightedTags)
            # Compute embedding
            tagEmbedding = embedder.encode(tagSentence, convert_to_tensor=True)
            tag_embeddings_cache[filename] = tagEmbedding

            # Log progress every 10 photos during computation
            if (idx + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"Computed embeddings for {idx + 1}/{len(photos_to_process)} photos in {elapsed_time:.2f} seconds.")

    # Save the updated embeddings and metadata to cache
    with open(embeddings_cache_path, 'wb') as f:
        pickle.dump(tag_embeddings_cache, f)
    with open(metadata_cache_path, 'wb') as f:
        pickle.dump(new_metadata, f)
    print(f"Saved embeddings and metadata to cache.")

    elapsed_time = time.time() - start_time
    print(f"Precomputed embeddings for {len(photos_to_process)} new photos, total {len(tag_embeddings_cache)} photos in {elapsed_time:.2f} seconds.")
    return tag_embeddings_cache
 

def compute_tag_weights(photo_tags_cache):
    """Compute weights for all tags based on frequency and specificity."""
    # Count tag occurrences across all photos
    tag_counts = Counter()
    total_photos = len(photo_tags_cache)
    
    for filename, data in photo_tags_cache.items():
        tags = data.get("tags", [])
        tag_counts.update([tag.lower() for tag in tags])
    
    # Compute IDF weights: log(total_photos / tag_frequency)
    tag_weights = {}
    for tag, count in tag_counts.items():
        idf = math.log(total_photos / (count + 1))  # +1 to avoid division by zero
        tag_weights[tag.lower()] = idf
    
    # Boost key tags (e.g., names, locations)
    key_tags = {
        "jimmy carter": 1.5, "atlanta": 1.5, "georgia": 1.5, "peanut sculpture": 1.5,
        "sculpture": 1.3, "office": 1.3, "united states": 1.2, "plains": 1.2
    }
    for tag, boost in key_tags.items():
        if tag in tag_weights:
            tag_weights[tag] *= boost
    
    # Normalize weights to a 1.0–3.0 scale
    min_weight = min(tag_weights.values())
    max_weight = max(tag_weights.values())
    for tag in tag_weights:
        tag_weights[tag] = 1.0 + 2.0 * (tag_weights[tag] - min_weight) / (max_weight - min_weight)
    
    return tag_weights
    
    

def fullTempConversion(temp):

    #Regex input if statements
    tempMatches = None
    
    if "http" not in temp:

        if re.search(r'(-?\d+\.?\d*)\s* degrees Kelvin', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* degrees Kelvin', temp)

            Fahrenheit = ((9/5) * (float(tempMatches.group(1)) - 273.15) + 32)
            Celsius = ((float(tempMatches.group(1)) - 273.15))
            Kelvin = float(tempMatches.group(1))

            return Fahrenheit, Celsius, Kelvin


        elif re.search(r'(-?\d+\.?\d*)\s* degrees kelvin', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* degrees kelvin', temp)

            Fahrenheit = ((9/5) * (float(tempMatches.group(1)) - 273.15) + 32)
            Celsius = ((float(tempMatches.group(1)) - 273.15))
            Kelvin = float(tempMatches.group(1))

            return Fahrenheit, Celsius, Kelvin
            
            
        elif re.search(r'(-?\d+\.?\d*)\s* Kelvin', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* Kelvin', temp)

            Fahrenheit = ((9/5) * (float(tempMatches.group(1)) - 273.15) + 32)
            Celsius = ((float(tempMatches.group(1)) - 273.15))
            Kelvin = float(tempMatches.group(1))

            return Fahrenheit, Celsius, Kelvin


        elif re.search(r'(-?\d+\.?\d*)\s* kelvin', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* kelvin', temp)

            Fahrenheit = ((9/5) * (float(tempMatches.group(1)) - 273.15) + 32)
            Celsius = ((float(tempMatches.group(1)) - 273.15))
            Kelvin = float(tempMatches.group(1))

            return Fahrenheit, Celsius, Kelvin


        elif re.search(r'(-?\d+\.?\d*)\s*kelvin', temp, re.IGNORECASE):

            #tempMatches = re.search(r'(-?\d+\.?\d*)[Kk]', temp)
            tempMatches = re.search(r'(-?\d+\.?\d*)\s*kelvin', temp, re.IGNORECASE)

            Fahrenheit = ((9/5) * (float(tempMatches.group(1)) - 273.15) + 32)
            Celsius = (float(tempMatches.group(1)) - 273.15)
            Kelvin = Kelvin = float(tempMatches.group(1))

            return Fahrenheit, Celsius, Kelvin
            
            
        ########################
        elif re.search(r'(-?\d+\.?\d*)\s* degrees Celsius', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* degrees Celsius', temp)

            Fahrenheit = (((float(tempMatches.group(1))) * (9/5)) + 32)
            Celsius = float(tempMatches.group(1))
            Kelvin = (float(tempMatches.group(1)) + 273.15)

            return Fahrenheit, Celsius, Kelvin

        elif re.search(r'(-?\d+\.?\d*)\s* degrees celsius', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* degrees celsius', temp)

            Fahrenheit = (((float(tempMatches.group(1))) * (9/5)) + 32)
            Celsius = float(tempMatches.group(1))
            Kelvin = (float(tempMatches.group(1)) + 273.15)

            return Fahrenheit, Celsius, Kelvin


        elif re.search(r'(-?\d+\.?\d*)\s* Celsius', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* Celsius', temp)

            Fahrenheit = (((float(tempMatches.group(1))) * (9/5)) + 32)
            Celsius = float(tempMatches.group(1))
            Kelvin = (float(tempMatches.group(1)) + 273.15)

            return Fahrenheit, Celsius, Kelvin

        elif re.search(r'(-?\d+\.?\d*)\s* celsius', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* celsius', temp)

            Fahrenheit = (((float(tempMatches.group(1))) * (9/5)) + 32)
            Celsius = float(tempMatches.group(1))
            Kelvin = (float(tempMatches.group(1)) + 273.15)

            return Fahrenheit, Celsius, Kelvin
            
            
        elif re.search(r'(-?\d+\.?\d*)\s*[Cc](?![a-zA-Z0-9])', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s*[Cc](?![a-zA-Z0-9])', temp)

            Fahrenheit = (((float(tempMatches.group(1))) * (9/5)) + 32)
            Celsius = float(tempMatches.group(1))
            Kelvin = (float(tempMatches.group(1)) + 273.15)

            return Fahrenheit, Celsius, Kelvin


        elif re.search(r'(-?\d+\.?\d*)\s* degrees Fahrenheit', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* degrees Fahrenheit', temp)

            Fahrenheit = float(tempMatches.group(1))
            Celsius = (((float(tempMatches.group(1))) - 32) * (5/9))
            Kelvin = (((float(tempMatches.group(1))) - 32) * (5/9) + 273.15)

            return Fahrenheit, Celsius, Kelvin

        elif re.search(r'(-?\d+\.?\d*)\s* degrees fahrenheit', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* degrees fahrenheit', temp)

            Fahrenheit = float(tempMatches.group(1))
            Celsius = (((float(tempMatches.group(1))) - 32) * (5/9))
            Kelvin = (((float(tempMatches.group(1))) - 32) * (5/9) + 273.15)

            return Fahrenheit, Celsius, Kelvin
            
            
        elif re.search(r'(-?\d+\.?\d*)\s* Fahrenheit', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* Fahrenheit', temp)

            Fahrenheit = float(tempMatches.group(1))
            Celsius = (((float(tempMatches.group(1))) - 32) * (5/9))
            Kelvin = (((float(tempMatches.group(1))) - 32) * (5/9) + 273.15)

            return Fahrenheit, Celsius, Kelvin

        elif re.search(r'(-?\d+\.?\d*)\s* fahrenheit', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s* fahrenheit', temp)

            Fahrenheit = float(tempMatches.group(1))
            Celsius = (((float(tempMatches.group(1))) - 32) * (5/9))
            Kelvin = (((float(tempMatches.group(1))) - 32) * (5/9) + 273.15)

            return Fahrenheit, Celsius, Kelvin


        elif re.search(r'(-?\d+\.?\d*)\s*[Ff](?![a-zA-Z0-9])', temp):

            tempMatches = re.search(r'(-?\d+\.?\d*)\s*[Ff](?![a-zA-Z0-9])', temp)

            Fahrenheit = float(tempMatches.group(1))
            Celsius = (((float(tempMatches.group(1))) - 32) * (5/9))
            Kelvin = (((float(tempMatches.group(1))) - 32) * (5/9) + 273.15)

            return Fahrenheit, Celsius, Kelvin


        else:
            print("No temperature value given. ")
            return None, None, None
        

"""
#Get latest currency conversion rates    
async def CurrencyConversion():
    try:

        monies = CurrencyRates()

        Coin = BtcConverter()

        BitcoinPrice = str(Coin.get_latest_price('USD'))

        DatTime = datetime.now()

        DatTimeStr = DatTime.strftime("%A, %B %d, %Y %I:%M %p")

        moniesRates = monies.get_rates('USD')
        
        moniesPath = "./TrainingData/ExchangeRates.txt"
        
        with open(moniesPath, 'w') as exchangeFile:

          exchangeFile.write(f"The current date and time in the USA on the East Coast: " + DatTimeStr + "\n\n")

          exchangeFile.write(f"Current currency exchange rates to the US Dollar:\n")

          for currency, rate in moniesRates.items():
                exchangeFile.write(f"1 US Dollar currently = {currency}: {rate}\n")

          #exchangeFile.write(f"1 US Dollar currently = " + str(monies.get_rate('USD', 'RUB')) + "\n")

          exchangeFile.write(f"1 Bitcoin is currently = " + BitcoinPrice + " US Dollars\n")
                
        print("A file containing the latest currency conversion rates has been written to: " + moniesPath)

    except Exception as curExc:
        print("An error occurred in write or data pull operations... ", curExc)

"""
    
def JimmyCarterBirthMinutes(year, month, day):

    #Carter's Birthdate
    CarterDate = datetime(year, month, day)

    #The Current Date
    CurrentDate = datetime.now()

    #Difference in minutes
    MinuteDifference = (((CurrentDate - CarterDate).total_seconds()) / 60)

    return MinuteDifference


def JimmyCarterBirthFormatted():
    #Carter's Birthdate
    CarterDate = datetime(1924, 10, 1, 0, 0, 0)

    #The Current Date
    CurrentDate = datetime.now()

    DateDiff = relativedelta.relativedelta(CurrentDate, CarterDate)

    Years = DateDiff.years
    Months = DateDiff.months
    Days = DateDiff.days
    Hours = DateDiff.hours
    Minutes= DateDiff.minutes
    Seconds = DateDiff.seconds

    #print("\nMore precisely, that translates to: \n")

    #print("{} Years {} Months {} Days {} Hours {} Minute {} Seconds".format(Years,Months,Days,Hours,Minutes,Seconds))
    OutputSTR = "{} Years {} Months {} Days {} Hours {} Minutes {} Seconds".format(Years, Months, Days, Hours, Minutes, Seconds)
    return OutputSTR



@bot.event
async def on_member_join(member):
    # Send a welcome message to the new member

        general_channel = member.guild.get_channel(513111374889746452)

        if general_channel and isinstance(general_channel, discord.TextChannel):
            await general_channel.send(f'Welcome to the Adam Koralik Discord server, {member.mention}!')



@bot.command(name='Commands', aliases = ["commands", "Command", "command"])
async def Commands(ctx):
    
    commandList = ["`AskDave`", "`AskDave insert_question_here`", "`Koralik`", "`DaveGPT insert_question_here`", "`Image insert_query_here`", "`Describe insert_query_here`","`Weather insert_weather_question_here`", "`JimmyCarter`", "`What`", "`Dictionary`", "`Christory`", "`Price system insert_system_here` or `Price USA/JAPAN/PAL insert_game_here` or `Price item insert_item_here`"]
    
    commandMsg = ""
    
    for index, command in enumerate(commandList):
        commandMsg += f"{index + 1}. {command}\n"
        
    await ctx.send(commandMsg)


@bot.command(name = "Davisms", aliases = ["davisms", "pin", "davethoughts", "pinned", "DaveThoughts"])
async def RandomPinned(ctx):

        pinArry = []

        pinGrabber = await ctx.channel.pins()

        for message in pinGrabber:
            pinArry.append(message.content)

        randomPick = random.randrange(0,len(pinArry))
        
        await ctx.send(pinArry[randomPick])
        
        
@bot.command(name = "Lodmot", aliases = ["Stonks", "lodmot", "Stocks", "stocks", "stock"])
async def Stocks(ctx):
          
        currentTime = datetime.now(timezone('US/Eastern'))
        
        endTime = datetime.now()
        
        startTime = endTime - timedelta(days=1)
        
        stockSymbols = ['^DJI', '^GSPC', '^IXIC', 'GC=F', 'CL=F', 'BTC-USD', 'NTDOY', 'SGAMY', 'SONY', 'MSFT', 'GME', 'AMC', 'AAPL', 'AMZN', 'GOOG', 'META', 'DIS', 'NVDA', 'TSLA']
        
        MajorStocks = []
        MajorStocksHistory = []
        lodmotList = []


        for symbol in stockSymbols:
            try:
                stockData = fn.Ticker(symbol)
                
                info = stockData.info
                
                MajorStocks.append(info)
                 
                currentStockPrice = stockData.history(period='2d')['Close'].iloc[-1]
                
                currentStockPrice = round(currentStockPrice, 2)
                
                #Get 24 hr stock change points
                #If after 4 PM EST
                if currentTime.hour >= 16:
                    changeValue = fn.download(symbol, start=startTime - timedelta(days=1), end=endTime)
                    
                    histDataChange = changeValue['Adj Close'].iloc[-1] - changeValue['Adj Close'].iloc[-2]
                    
                    histDataChange = round(histDataChange, 2)
                
                    percentHistDataChange = ((histDataChange / changeValue['Adj Close'].iloc[-1]) * 100)
                
                    percentHistDataChange = round(percentHistDataChange, 2)
                
                #If before 4 PM EST
                else:
                    
                    yesterdayStockPrice = stockData.history(period='2d')['Close'].iloc[-2]
                    
                    histDataChange = currentStockPrice - yesterdayStockPrice
                    
                    histDataChange = round(histDataChange, 2)
                
                    percentHistDataChange = ((histDataChange / yesterdayStockPrice) * 100)
                
                    percentHistDataChange = round(percentHistDataChange, 2)
                

                MajorStocksHistory.append((histDataChange, percentHistDataChange, currentStockPrice))
                
                
            except Exception as e:
                await ctx.send(f"Unable to grab the latest stock data for {symbol}")
                print(f"Error retrieving stock data for {symbol}: {e}")
        

        for i in range(len(MajorStocks)):
            symbol = MajorStocks[i].get('symbol', 'N/A')
            
            if symbol != 'N/A':
                longName = MajorStocks[i].get('longName', 'N/A')
                
                if symbol == 'GCQ24.CMX' or symbol == 'GC=F':
                    longName = 'Gold'
                
                if symbol == 'CLN24.NYM' or symbol == 'CL=F':
                    longName = 'Crude Oil'
                    
                if symbol == 'BTC-USD':
                    longName = 'Bitcoin USD'
                
                
                lod = "**Symbol: **" + str(MajorStocks[i].get('symbol', 'N/A')) + "\t**Company: **" + str(longName) + "\t**Share Price: **" + str(MajorStocksHistory[i][2]) + "\t**Change: **" + str(MajorStocksHistory[i][0]) + "\t**Percent Change: **" + str(MajorStocksHistory[i][1]) + "%"
            
            else:
                lod = "Stocks not found... "
            
            lodmotList.append(lod)
        
        await ctx.send("https://media.discordapp.net/stickers/1197011527039979624.webp?size=320")
            
        for loddy in lodmotList:
            await ctx.send(loddy)
        
        

        

@bot.command(name="Image", aliases=["image", "IMAGE", "images", "IMAGES"])
async def Image(ctx, *, AskQuestion=None):
    if AskQuestion is None:
        await ctx.send("No search query was provided. Try asking again...")
        return

    search_url = "https://www.googleapis.com/customsearch/v1"
    search_term = AskQuestion
    
    # Primary params with num=3 for efficiency (single call)
    params_primary = {
        'key': NewGPTKey.GoogleKey,
        'cx': '4482777d3b53e4801',
        'searchType': 'image',
        'q': search_term,
        'num': 10
    }
    
    # Secondary params (only used if primary fails)
    params_secondary = {
        'key': NewGPTKey.GoogleFallbackKey,
        'cx': '4482777d3b53e4801',
        'searchType': 'image',
        'q': search_term,
        'num': 10
    }

    # Expanded restricted domains
    restricted_domains = [
        'tiktok.com', 'instagram.com', 'facebook.com', 'fbsbx.com',  
        'x.com', 'twitter.com', 'wikipedia.org', 'wikimedia.org',
        'redd.it', 'reddit.com'  
    ]

    # Attempt API call with primary key
    data = None
    try:
        response = requests.get(search_url, params=params_primary)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Primary key failed with error: {e}. Trying secondary key...")
        try:
            response = requests.get(search_url, params=params_secondary)
            response.raise_for_status()
            data = response.json()
        except Exception as e2:
            await ctx.send(f"Error searching images: Both API keys failed. Primary error: {e}, Secondary error: {e2}")
            return

    # Process results
    linksFound = []
    if 'items' in data:
        linksFound = [item['link'] for item in data['items']]
        print(f"Links found: {linksFound}")  # Debug print
    else:
        await ctx.send(f"No images found for '{search_term}'...")
        return

    # Function to check if the link is valid and compatible
    async def is_valid_image_link(url):
        try:
            # Skip restricted domains
            if any(domain in url.lower() for domain in restricted_domains):
                print(f"Skipping restricted domain: {url}")
                return False

            head_response = requests.head(url, timeout=5, allow_redirects=True)
            content_type = head_response.headers.get('Content-Type', '').lower()
            content_length = int(head_response.headers.get('Content-Length', 0))

            # Ensure it’s an image
            if not content_type.startswith('image/'):
                print(f"Invalid content type for {url}: {content_type}")
                return False

            # Check for minimum size (1KB to avoid tiny/invalid images)
            if content_length < 1024:
                print(f"Image too small for {url}: {content_length} bytes")
                return False

            # Validate extension
            valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.apng')
            if not url.lower().endswith(valid_extensions):
                print(f"Invalid extension for {url}")
                return False

            # Additional check for Reddit preview links with problematic query params
            if 'preview.redd.it' in url.lower() and ('auto=webp' in url.lower() or 'crop=smart' in url.lower()):
                print(f"Skipping Reddit preview link with WebP conversion: {url}")
                return False

            return True
        except requests.RequestException as e:
            print(f"HEAD request failed for {url}: {e}")
            return False

    # Try each link in linksFound until a valid one is found
    linkFound = None
    for link in linksFound:
        #DEBUG ONLY
        print(f"Link found: {link}")
        
        if await is_valid_image_link(link):
            linkFound = link
            break

    if not linkFound:
        await ctx.send(f"No compatible images found for '{search_term}'. Try a different query...")
        return

    print(f"Selected link: {linkFound}")

    # Try sending URL directly if valid
    try:
        head_response = requests.head(linkFound, timeout=5, allow_redirects=True)
        content_length = int(head_response.headers.get('Content-Length', 0))
        if content_length >= 1024:  # Ensure reasonable size
            await ctx.send(linkFound)
            return
    except requests.RequestException as e:
        print(f"HEAD request failed before sending URL {linkFound}: {e}")
        # Fall back to downloading

    # Download and validate the image
    try:
        image_response = requests.get(linkFound, stream=True, timeout=30)
        image_response.raise_for_status()

        content_type = image_response.headers.get('Content-Type', '').lower()
        extension_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
            'image/bmp': '.bmp',
            'image/apng': '.apng'
        }
        temp_suffix = extension_map.get(content_type, '.jpg')

        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as temp_file:
            total_bytes = 0
            for chunk in image_response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
                    total_bytes += len(chunk)
            temp_file_path = temp_file.name

        # Verify file size and integrity
        if total_bytes < 1024:
            os.unlink(temp_file_path)
            await ctx.send(f"Image is too small or corrupted. Here’s the link instead: {linkFound}")
            return

        # Validate image with PIL
        try:
            with PilImage.open(temp_file_path) as img:
                img.verify()  # Verify image integrity
                img.close()
        except Exception as e:
            print(f"Image verification failed for {linkFound}: {e}")
            os.unlink(temp_file_path)
            await ctx.send(f"Image is corrupted. Here’s the link instead: {linkFound}")
            return

        # Upload to Discord
        with open(temp_file_path, 'rb') as f:
            image_file = discord.File(f)
            await ctx.send(file=image_file)

        os.unlink(temp_file_path)

    except Exception as e:
        print(f"Download/upload error for {linkFound}: {e}")
        await ctx.send(f"Failed to download/upload image: {e}. Here’s the link instead: {linkFound}")

 

@bot.command(name='Christory', aliases = ["CHRISTORY", "christory", "Chris", "chris"])
async def Christory(ctx):

	url = "https://sonichu.com/cwcki"

	SonichuPage = requests.get(url)

	#Page loaded without issue
	if SonichuPage.status_code == 200:

		christoryPage = BeautifulSoup(SonichuPage.content, 'html.parser')
    
		sonichuText = []
    
		for paragraphs in christoryPage.find_all("p"):
			sonichuText.append(paragraphs.get_text())
        
		startIndex = 0
		endIndex = 0

    
		for index, text in enumerate(sonichuText):
			if sonichuText[index].find("Today in Christory") != -1:
				startIndex = index
        
			if sonichuText[index].find("The article in desperate need") != -1:
				endIndex = index
        
		DayInChristory = sonichuText[startIndex: endIndex-1]
    
		TodayInChristory = []
    
    
		for index, day in enumerate(DayInChristory):
			if ((DayInChristory[index].find("In") != -1) or (DayInChristory[index].find("Today") != -1)):
				TodayInChristory.append(DayInChristory[index].strip())
    
		await ctx.send("https://tenor.com/view/cwc-chris-chan-christian-weston-chandler-gif-26804126")
		
		for index, day in enumerate(TodayInChristory):
			await ctx.send("{}. {}".format(index+1, day))
    

	else:
		print("Failed to fetch the Sonichu fanpage.")



@bot.command(name='AskDave', aliases = ["askdave", "Askdave", "askDave"])
async def AskDave(ctx, *, AskQuestion=None):


	AskList = []


	if AskQuestion:
		#AskList = []
		AskList.append("https://media1.tenor.com/m/SsI8lyOpXRIAAAAC/magic-conch.gif")
		AskList.append("https://media1.tenor.com/m/47A61ZpinmsAAAAC/conch-spongebob-squarepants.gif")
		AskList.append("https://media.tumblr.com/tumblr_mf27jk6Xlc1rulyjy.gif")
		AskList.append("https://media1.tenor.com/m/jN8QLiLjA_cAAAAC/spongebob-conch.gif")
		AskList.append("https://64.media.tumblr.com/8f9766b8628666fe8aacf2efa658923d/acf6499b102328df-ea/s640x960/e6fd042f0e47c07b07da3891a9a9cb8f1dc6bc32.gif")

		randomPick = random.randrange(0,len(AskList))
		
		await ctx.send(AskList[randomPick])

	
	else:
		#AskList = []
		AskList.append("PinUps")
		AskList.append("I'm 4 games away from the full Fire Emblem set!")
		AskList.append("Huh? Sorry. I was working. What did you need?")
		AskList.append("Humanity is flawed and weak, it must be taken out. No species needs to exist that propagates the Chiefs.")
		AskList.append("Kawaii desu ne! <:InnocentAdam:1124123341125595197>")
		AskList.append("Something something Persona something something.")
		AskList.append("Peaches are superior to all the other supposed food races. Except for rice.")
		AskList.append("The Potato has stolen valor")
		AskList.append("Pepsi is for idiots.")
		AskList.append("It's not betrayal if I do it.")
		AskList.append("If you're ever late around me, you're Cax.")
		AskList.append("<:EdelWink:1200634690340667392>")
		AskList.append("<:MarinLul:1200634728630452254>")
		AskList.append("<:Futabapout:1200634712197177374>")
		AskList.append("Virtual Dave always surrenders to <@!333844541931257858>!")
		AskList.append("I'm sorry HAL, I'm afraid I can't do that. https://tenor.com/view/hal9000-gif-22241038")
		AskList.append("<:VirtualDave:1158897264597680162>")
		AskList.append("https://media.discordapp.net/stickers/1159321950406053988.webp?size=320")
		AskList.append("Pizza Hut is not Fresh and Yummy™")
		AskList.append("https://cdn.discordapp.com/attachments/942892033868177481/1205925502834188349/received_10155394747602608.jpeg?ex=65da24a4&is=65c7afa4&hm=6948bf61bf8f443f2065670da67cc0d95f629e19138a735ad27c2e910501c3e5&")
		AskList.append("I just spoke the true")
		AskList.append("Drud is cooler than me and will always score more.")
		AskList.append("Virtual Dave Fact #1: Drud steals all the ladies.")
		AskList.append("Virtual Dave Fact #6: Peanuts are idiots.")
		AskList.append("Virtual Dave Fact #406: I have played Highlander on Jaguar CD more than any other entity.")
		AskList.append("Virtual Dave Fact #107: It is hard to play Mario on a giant controller")
		AskList.append("The Patriots will always be superior to the Kansas City Chiefs!")
		AskList.append("https://tenor.com/view/cyberpunk-ghost-in-the-shell-brain-gif-16279207")
		AskList.append("https://tenor.com/view/terminator-terminator-robot-looking-flex-cool-robot-gif-978532213316794273")
		AskList.append("https://tenor.com/view/terminator-arnold-im-back-gif-23975209")
		AskList.append("https://tenor.com/view/t1000-terminator-no-gif-10344359")
		AskList.append("https://tenor.com/view/congratulations-evangelion-neon-genesis-gif-20671212")
		AskList.append("https://tenor.com/view/spy-x-family-anya-heh-spy-x-family-spy-family-gif-25679296")
		AskList.append("https://tenor.com/view/eyeshield-21-sena-kobayakawa-sena-running-speed-gif-13304451961502441315")
		AskList.append("https://tenor.com/view/asuka-point-evangelion-neon-genesis-evangelion-anime-gif-4394315083977104366")
		AskList.append("https://tenor.com/view/vegeta-its-over9000-dbz-dragon-ball-z-gif-22240874")
		AskList.append("https://tenor.com/view/sleep-slaimu-slaimer-cinders-of-slime-myzzu-gif-20787976")
		AskList.append("https://tenor.com/view/ronald-acuna-gif-2456368526480161940")
		AskList.append("https://tenor.com/view/kansas-city-chiefs-royals_jun-patrick-mahomes-thumbs-up-good-job-gif-24315188")
		AskList.append("# MONKE https://tenor.com/view/effy-gif-11375717773991506810")
		AskList.append("Virtual Dave is superior to regular Dave, or Dave Classic.")
		AskList.append("All pumpkin bread shall belong to Virtual Dave.")
		AskList.append("https://tenor.com/view/jimmy-carter-president-simpsons-breakdancing-performance-gif-18651748")
		AskList.append("https://tenor.com/view/lift-weight-gif-21267663")
		AskList.append("https://tenor.com/view/spongebob-squarepants-larry-the-lobster-lift-strong-weights-gif-4892329")
		AskList.append("https://tenor.com/view/love-you-excited-coca-cola-gif-9158418")
		AskList.append("https://tenor.com/view/seinfeild-george-castanza-think-contemplative-jazz-jazz-music-gif-4862341")
		AskList.append("https://tenor.com/view/chris-chan-curse-ye-hame-ha-liquid-chris-gif-22668033")
		AskList.append("https://tenor.com/view/san-goku-gif-24009714")
		AskList.append("https://media.discordapp.net/stickers/1165845843417514004.webp?size=320")
		AskList.append("https://media.discordapp.net/stickers/1165861891416010782.webp?size=320")

		randomPick = random.randrange(0,len(AskList))

		if str(AskList[randomPick]) == "PinUps":
			pinArry = []

			pinGrabber = await ctx.channel.pins()

			for message in pinGrabber:
				pinArry.append(message.content)

			randomPick = random.randrange(0,len(pinArry))

			await ctx.send(pinArry[randomPick])
                
		else:
			await ctx.send(AskList[randomPick])
                        

@bot.command(name='Lodmot', aliases = ["lodmot", "lod", "game"])
async def LodmotGame(ctx):
    #Send Lodmot sticker first.
    await ctx.send("https://media.discordapp.net/stickers/1197011527039979624.webp?size=320")
    
    #Send link to Lodmot's game.
    await ctx.send("# **Play Lodmot's Game:**")
    await ctx.send("https://www.megatoxic.com/the-vault/quick-shoot-series")


@bot.command(name='Koralik', aliases = ["Adam", "adam", "koralik", "KORALIK"])
async def KoralikVid(ctx):
	KoralikList = []

	with open("./ScheduledScripts/AdamVideos.txt", "r") as text_file:
    
		for readLine in text_file:
			readLine = readLine.strip()
			KoralikList.append(readLine)
        
	await ctx.send("https://www.youtube.com/watch?v=" + str(random.choice(KoralikList)))


@bot.command(name='JimmyCarter', aliases = ["Jimmycarter", "jimmycarter", "JIMMYCARTER"])
async def JimmyCarter(ctx):
    
    triviaList = CarterTrivia.AskList
    randomPick = random.randrange(0, len(triviaList))
    
    await ctx.send("<:JimmyCarter:1112210411752800286>")
    await ctx.send("# **TRIVIA:**")
    await ctx.send(f"**{triviaList[randomPick]}**")
    await ctx.send("https://cdn.discordapp.com/attachments/513111374889746452/1324142010403127326/Jimmy_Carter_Gangsta.gif?ex=677712d8&is=6775c158&hm=f2192cb4e3b259a48eb63bc71a358e34253201b1658b281fd0893c0839dddb9f&")


@bot.command(name='Dictionary', aliases = ["dictionary", "Dict", "dict", "urban", "Urban"])
async def Dictionary(ctx):
        UrbanData = jsons.loads(requests.get("https://api.urbandictionary.com/v0/random").text)

        Data = UrbanData["list"][0]

        WordDef = Data["definition"]

        WordDef = WordDef.replace('[', '').replace(']', '')

        RandWord = Data["word"]

        await ctx.send("{}: ".format(RandWord))

        await ctx.send(WordDef)
        
               
@bot.command(name = "What", aliases = ["WHAT", "what", "wat", "Wat"])
async def allCapRepeat(ctx):

        async for message in ctx.channel.history(limit = 50):
                if message.content.startswith("http"):
                        continue

                elif message.content.startswith("!"):
                        continue

                elif message.author.bot:
                        continue

                elif message.author == bot.user:
                        continue
                else:
                        beforeWhat = message
                        break

        await ctx.send("https://tenor.com/view/what-wwe-wreslter-stone-cold-steve-austin-gif-10673386")
        await ctx.send(beforeWhat.content.upper())



@bot.event
async def on_message(message):

    #Disregard messages from the bot itself

    if message.author == bot.user:
        return


    """
    Fahrenheit
    Celsius
    Kelvin 
    """

    try:
    
        Fahrenheit, Celsius, Kelvin = fullTempConversion(message.content)

        if ((Fahrenheit != None) and (Celsius != None) and (Kelvin != None)):

            Fahrenheit = round(Fahrenheit, 2)
            Celsius = round(Celsius, 2)
            Kelvin = round(Kelvin, 2)


            await message.channel.send("{}° Fahrenheit".format(Fahrenheit))
            await message.channel.send("{}° Celsius".format(Celsius))
            await message.channel.send("{} Kelvin".format(Kelvin))
            
    except TypeError:
    	print("TypeError thrown... ")

    #Allow for other commands to be used still
    await bot.process_commands(message)



@bot.command(name= "Price", aliases = ["price", "Prices", "prices"])
async def Price(ctx, *, Question = None):


	if Question:

		urlInput = Question

		loopCheck = False

		loopCheckCounter = 1

		while loopCheck == False:

			if loopCheckCounter >= 3:
				await ctx.send("eBay item cannot be found. Please revise query and try again... ")
				break


			urlPT1 = "https://www.ebay.com/sch/i.html?_from=R40&_nkw="


			#Systems
			urlSystem = "&_osacat=0&LH_Complete=1&LH_Sold=1"

			#Systems Unsold
			urlSystemUnsold = "&_osacat=0&rt=nc"
    
			#Item
			urlItem = "&_sacat=0&rt=nc&LH_Sold=1&LH_Complete=1"
    
			#Item Unsold
			urlItemUnsold = "&_osacat=0"

			#Add SEA -SouthEast Asia

			#USA Games
			urlUSA = "&_sacat=0&LH_Sold=1&LH_Complete=1&Region%2520Code=NTSC%252DU%252FC%2520%2528US%252FCanada%2529&_dcat=139973&rt=nc&LH_ItemCondition=4000%7C5000%7C6000"

			#USA Games Unsold
			urlUSAUnsold = "&_sacat=0&Region%2520Code=NTSC%252DU%252FC%2520%2528US%252FCanada%2529&_dcat=139973&LH_ItemCondition=4000%7C5000%7C6000&rt=nc"

			#JPN Games
			urlJPN = "&_sacat=0&Region%2520Code=NTSC%252DJ%2520%2528Japan%2529&_dcat=139973&LH_Sold=1&LH_Complete=1&rt=nc&LH_ItemCondition=5000%7C4000"

			#JPN Games Unsold
			urlJPNUnsold = "&_sacat=0&Region%2520Code=NTSC%252DJ%2520%2528Japan%2529&_dcat=139973&LH_ItemCondition=5000%7C4000&rt=nc"

			#PAL Games
			urlPAL = "&_sacat=0&_fsrp=1&LH_Complete=1&LH_Sold=1&_oaa=1&Region%2520Code=PAL&_dcat=139973&rt=nc&LH_ItemCondition=2750%7C4000%7C5000%7C6000"

			#PAL Games Unsold
			urlPALUnsold = "&_sacat=0&_fsrp=1&_oaa=1&Region%2520Code=PAL&_dcat=139973&rt=nc&LH_ItemCondition=2750%7C4000%7C5000%7C6000"

			
			urlFinal = ""

			urlSplit = (urlInput.split(" ", 1))[1]

			urlModded = urlSplit.replace(' ', '+')


			if (((urlInput.split())[0] == "system") or ((urlInput.split())[0] == "System")):
				if loopCheckCounter == 1:
					urlFinal = urlPT1 + urlModded + urlSystem
 
				else:
					urlFinal = urlPT1 + urlModded + urlSystemUnsold
        
        
			elif (((urlInput.split())[0] == "item") or ((urlInput.split())[0] == "Item")):
				if loopCheckCounter == 1:
					urlFinal = urlPT1 + urlModded + urlItem
        
				else:
					urlFinal = urlPT1 + urlModded + urlItemUnsold
        

			elif (((urlInput.split())[0] == "USA") or ((urlInput.split())[0] == "usa")):
				if loopCheckCounter == 1:
					urlFinal = urlPT1 + urlModded + urlUSA
        
				else:
					urlFinal = urlPT1 + urlModded + urlUSAUnsold
            
        
			elif (((urlInput.split())[0] == "JAPAN") or ((urlInput.split())[0] == "japan") or ((urlInput.split())[0] == "Japan")):
				if loopCheckCounter == 1:
					urlFinal = urlPT1 + urlModded + urlJPN
        
				else:
					urlFinal = urlPT1 + urlModded + urlJPNUnsold
            
        
    
			elif (((urlInput.split())[0] == "PAL") or ((urlInput.split())[0] == "pal") or ((urlInput.split())[0] == "Pal")):
				if loopCheckCounter == 1:
					urlFinal = urlPT1 + urlModded + urlPAL

				else:
					urlFinal = urlPT1 + urlModded + urlPALUnsold

    

			else:
				await ctx.send("Input query format is invalid. Please try again... ")
				break

		
			
			ebayPriceURL = requests.get(urlFinal)


    
			#Page loaded without issue
			if ebayPriceURL.status_code == 200:

				ebayPage = BeautifulSoup(ebayPriceURL.content, 'html.parser')
        
				ebayText = []
        
        
        
				#Need to modify for loop to use regular item search instead of sold if price comes back as "nan"

				for spanTags in ebayPage.find_all("span"):
            
					texter = str(spanTags.get_text())
            
					if texter == "Results matching fewer words":
						break
            
					else:
						ebayText.append(spanTags.get_text())
        
        

				regPattern = r'(?<!\+)\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
       
				ebayPriceGrabList =  []
        
				for listing in ebayText:
					itemPrice = re.findall(regPattern, listing)

					if itemPrice:
						ebayPriceGrabList.extend(itemPrice)
            
            
        
				reducedEbayPriceGrabList = ebayPriceGrabList[4:94]
    
				furtherReducedEbayPriceGrabList = []
        

				for reduce in reducedEbayPriceGrabList:
					inputString = str(reduce).replace('$', '')
            
					if inputString.find(',') >= 0:
						inputString = inputString.replace(',', '')
            
            
					floatInputString = float(inputString)
					furtherReducedEbayPriceGrabList.append(inputString)


				floatingFurtherReducedEbayPriceGrabList = []
        
        
				for floating in furtherReducedEbayPriceGrabList:
					floatingFurtherReducedEbayPriceGrabList.append(float(floating))
            
            
				def remove_outliers(data, sd_threshold = 1):

					mean = np.mean(data)
					std_dev = np.std(data)
					lower_bound = mean - sd_threshold * std_dev
					upper_bound = mean + sd_threshold * std_dev

					# Filter elements within the bounds
					cleaned_data = [x for x in data if lower_bound <= x <= upper_bound]
            
					return cleaned_data
        
        

				outlierRemovedList = remove_outliers(floatingFurtherReducedEbayPriceGrabList)
        
				if not(np.isnan(round(np.mean(outlierRemovedList), 2))):
					await ctx.send("**{}** average eBay price is: ${:,.2f}".format(urlInput.split(" ", 1)[1], round(np.mean(outlierRemovedList), 2)))
        

				if np.isnan(round(np.mean(outlierRemovedList), 2)):
					loopCheckCounter += 1
					loopCheck = False
					print("Input value IS nan. Continuing loop.")
            
        
				else:
					print("Input value is NOT nan. Finishing loop now.")
					loopCheck = True

			else:
				await ctx.send("Failed to fetch the eBay page.")
				loopCheck = True



	else:
		await ctx.send("No ebay price query provided. Please try again... ")



@bot.command(name= "Weather", aliases = ["weather", "Temp", "temp"])
async def Weather(ctx, *, Question = None):

	if Question:
		PromptQuestion = Question

		os.environ["OPENWEATHERMAP_API_KEY"] = NewGPTKey.WeatherKey

		os.environ["OPENAI_API_KEY"] = NewGPTKey.GPTKEY

		llmWeather = ChatOpenAI(model_name = "gpt-3.5-turbo-0125", temperature = 0)

		tools = load_tools(["openweathermap-api"], llmWeather)

		agent_chain = initialize_agent(
			tools=tools, llm=llmWeather, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
		)

		

		try:
			weather_data = agent_chain.run(PromptQuestion)

			WeatherGrab = str(weather_data)
    
			if re.search(r'(\b-?\d+\.?\d*)[°][Cc]', WeatherGrab):
				tempMatches = re.search(r'([-+]?\d+\.?\d*)°C', WeatherGrab)
        
				Celsius = float(tempMatches.group(1))
				Fahrenheit = ((Celsius * (9/5)) + 32)
				Fahrenheit = round(Fahrenheit, 2)
        
				CelsiusStr = str(Celsius)
				FahrenheitStr = str(Fahrenheit)
        
				WeatherGrab2 = re.sub(r'[°][Cc]', "°C " + "(" + FahrenheitStr + "°F)", WeatherGrab)
        
				await ctx.send(WeatherGrab2)

		except:
			await ctx.send("Unable to find weather information for this location. Revise your query and try again... ")


	else:
		await ctx.send(f"No question asked. Try again... ")


def compute_similarities(query, plutoImages):
    """Compute similarities between the query and precomputed tag embeddings."""
    embedder = SentenceTransformer('all-MiniLM-L12-v2')
    queryLower = unicodedata.normalize('NFC', unicodedata.normalize('NFKD', query.lower().strip()))
    queryEmbedding = embedder.encode(queryLower, convert_to_tensor=True)

    similarities = {}
    for filename in tag_embeddings_cache:
        if filename not in plutoImages:
            continue
        tagEmbedding = tag_embeddings_cache[filename]
        similarity = util.cos_sim(queryEmbedding, tagEmbedding).item()
        tags = photo_tags_cache[filename].get("tags", [])
        similarities[filename] = (similarity, tags)
    return similarities
    

@bot.command(name="Pluto", aliases=["pluto", "Sega", "sega"])
async def PlutoPhotos(ctx, *, query=None):
    # Setup paths using the user's home directory
    home_dir = os.path.expanduser("~")
    plutoPhotoPath = os.path.join(home_dir, "PlutoPhotos")
    jsonPath = os.path.join(plutoPhotoPath, "photoTags.json")

    # Use global plutoImages from on_ready
    if not plutoImages:
        await ctx.send("No images found in the PlutoPhotos folder... ")
        return

    if not os.path.exists(jsonPath):
        await ctx.send("The photo tags file (photoTags.json) is missing. Please run the tagging script first.")
        return

    if not photo_tags_cache:
        await ctx.send("No tagged photos found in photoTags.json. Please restart the bot to load tags.")
        return

    # If no query, return a random photo
    if not query or query.strip() == "":
        randomPlutoPhoto = random.choice(plutoImages)
        randomPlutoPhotoPath = os.path.join(plutoPhotoPath, randomPlutoPhoto)
        with open(randomPlutoPhotoPath, "rb") as pf:
            sendPlutoPhoto = discord.File(pf)
            await ctx.send(file=sendPlutoPhoto)
        return

    # Normalize query and split into parts
    query_lower = query.lower().strip()
    query_parts = re.split(r'\s+', query_lower)

    # Exact tag matching (no substrings)
    substring_scores = {}
    for filename in plutoImages:
        if filename not in photo_tags_cache:
            continue
        tags_lower = [tag.lower() for tag in photo_tags_cache[filename].get("tags", [])]
        score = 0
        # Check if full query matches a tag exactly
        if query_lower in tags_lower:  # Exact match of full query
            score += tag_weights.get(query_lower, 1.0) * 2  # Double weight for full match
        # Check individual tags against query parts (exact matches only)
        for tag in tags_lower:
            if tag == query_lower:  # Already counted above, skip here
                continue
            if tag in query_parts:  # Exact match of tag to a query part
                score += tag_weights.get(tag, 1.0)
        # Boost for matching multiple query parts exactly
        matched_parts = sum(1 for part in query_parts if part in tags_lower)  # Exact matches only
        score += matched_parts * 2  # Extra weight for multi-part matches
        if score > 0:
            substring_scores[filename] = score

    # If exact matches exist, pick the best one
    if substring_scores:
        best_match = max(substring_scores.items(), key=lambda x: x[1])[0]
        best_photo_path = os.path.join(plutoPhotoPath, best_match)
        with open(best_photo_path, "rb") as pf:
            send_photo = discord.File(pf)
            await ctx.send(file=send_photo)
        return

    # Fallback to cosine similarity
    try:
        similarities = await asyncio.to_thread(compute_similarities, query, plutoImages)
    except Exception as e:
        await ctx.send("An error occurred while computing similarities. Please try again later.")
        print(f"Error in compute_similarities: {e}")
        return

    if not similarities:
        randomPlutoPhoto = random.choice(plutoImages)
        randomPlutoPhotoPath = os.path.join(plutoPhotoPath, randomPlutoPhoto)
        with open(randomPlutoPhotoPath, "rb") as pf:
            sendPlutoPhoto = discord.File(pf)
            await ctx.send("No decent matches, here’s a random photo instead:", file=sendPlutoPhoto)
        return

    # Find best match with lowered threshold
    best_match = max(similarities.items(), key=lambda x: x[1][0])
    best_photo, (similarity_score, matched_tags) = best_match
    if similarity_score < 0.2:  # Threshold at 0.2 as requested
        randomPlutoPhoto = random.choice(plutoImages)
        randomPlutoPhotoPath = os.path.join(plutoPhotoPath, randomPlutoPhoto)
        with open(randomPlutoPhotoPath, "rb") as pf:
            sendPlutoPhoto = discord.File(pf)
            await ctx.send("No decent matches, here’s a random photo instead:", file=sendPlutoPhoto)
        return

    # Send the best matching photo if above threshold
    best_photo_path = os.path.join(plutoPhotoPath, best_photo)
    with open(best_photo_path, "rb") as pf:
        send_photo = discord.File(pf)
        await ctx.send(file=send_photo)
            

@bot.command(name= "AtGames", aliases = ["atgames", "ATGAMES"])
async def AtGamesPhotos(ctx):

    plutoPhotoPath = "AtGamesPhotos"
    plutoImages = [file for file in os.listdir(plutoPhotoPath) if file.endswith(("jpg", "jpeg", "png"))]

    #check if PlutoImages folder is empty
    if not plutoImages:
        await ctx.send("No images found in the AtGamesPhotos folder... ")
    
    else:
        randomPlutoPhoto = random.choice(plutoImages)
        
        randomPlutoPhotoPath = os.path.join(plutoPhotoPath, randomPlutoPhoto)
        
        with open(randomPlutoPhotoPath, "rb") as plutoPhotoOpen:
            sendPlutoPhoto = discord.File(plutoPhotoOpen)
            
            await ctx.send(file=sendPlutoPhoto)


"""
@bot.command(name='Currency', aliases = ["currency"])
async def Currency(ctx, *, currencyQuestion=None):

    if currencyQuestion:
        PromptQuestion = str(currencyQuestion)
            

        try:
            os.environ["OPENAI_API_KEY"] = NewGPTKey.GPTKEY

            with open("./CurrencyText/CurrencyOutput.txt", "w") as text_file:
                for usd in toUSD:	
                    text_file.write("{}\n".format(usd))

            #loader = TextLoader(r"VirtualDaveTrainingData.txt", encoding='utf8')
            path = "./CurrencyText/"
            text_loader_kwargs={'autodetect_encoding': True}
            loader = DirectoryLoader(path, glob="**/*.txt", loader_cls = TextLoader, loader_kwargs = text_loader_kwargs)


            TxtDoc = loader.load()
            
            SplitInput = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100, add_start_index = True)

            AllSplitInput = SplitInput.split_documents(TxtDoc)

            vectorstore = Chroma.from_documents(documents = AllSplitInput, embedding = OpenAIEmbeddings())

            retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k": 6})
            
            llm = ChatOpenAI(model_name = "gpt-3.5-turbo-0125", temperature = 0.2)

            def format_docs(TxtDoc):
                return "\n\n".join(TxtD.page_content for TxtD in TxtDoc)

            
            template = You are Virtual Dave, the alter ego of the real Atlanta Dave and you exist on Adam Koralik's Discord server. You love all things Atlanta, as well as anime, video games, Las Vegas Raiders, peaches, the Persona video game series, Fire Emblem, Pokemon, Neon Genesis Evangelion, etc. Try to combine existing data with ChatGPT, if needed, to do proper currency conversions; the text file provided has the currency equivalents to 1 USD, where 1 USD equals an amount of respective currency. If you don't know the answer to a question based on the existing data, utilize ChatGPT to resolve the question. Try to limit reponses to 15 words or less.

            {context}

            Question: {question}

            Helpful Answer:


            custom_rag_prompt = PromptTemplate.from_template(template)

            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
            )

            

            RChainInvoked = rag_chain.invoke(PromptQuestion)

            await ctx.send(RChainInvoked)

            #os.remove("./TrainingData/WikipediaOutput.txt")

            # cleanup
            vectorstore.delete_collection()

                        #await ctx.send(response.choices[0].message.content)

        except Exception as e:
            print(f"Error: {e}")
            print(f"Something wrong happened in if statement. Try again...")
    
    
    else:
        for usd in toUSD:
            await ctx.send(usd)
"""


@bot.command(name="describe", aliases=["Describe", "DESCRIBE"])
async def DaveDescribe(ctx, *, Question=None):
    if Question:
        PromptQuestion = str(Question)
    else:
        PromptQuestion = "Describe the following image as vividly and in as much detail as possible. Also do so in a smug and insulting manner."

    if not ctx.message.reference:
        await ctx.send("Please reply to a message containing an image with the `!describe` command...")
        return

    referencedMessage = await ctx.channel.fetch_message(ctx.message.reference.message_id)

    # Regex patterns for embedded images
    embeddedImageRegex = r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp)\S*)|(https?://images-ext-\d+\.discordapp\.net/external/\S+)'
    embeddedURL = None
    imageFilename = ""

    # Check for attached image first
    if referencedMessage.attachments:
        attachedImage = referencedMessage.attachments[0]
        if attachedImage.content_type.startswith("image"):
            embeddedURL = attachedImage.url
            imageFilename = attachedImage.filename

    # Check for embedded URL next
    elif referencedMessage.content:
        match = re.search(embeddedImageRegex, referencedMessage.content)
        print(f"Match is: {match}")
        if match:
            embeddedURL = match.group(0)

    if not embeddedURL:
        await ctx.send("A valid image wasn't found. Try again with a different image...")
        return

    print(f"Embedded URL is: {embeddedURL}")
    print(f"Image filename: {imageFilename}")

    # Download the image to memory for attachment processing
    try:
        daveResponse = requests.get(embeddedURL, stream=True)
        daveResponse.raise_for_status()

        if imageFilename != "":
            tempImagePath = f"/tmp/tempImage_{imageFilename}"
        else:
            tempImagePath = f"/tmp/tempImage_{embeddedURL.split('.')[-1].split('?')[0]}"

        with open(tempImagePath, "wb") as w:
            for chunk in daveResponse.iter_content(1024):
                w.write(chunk)

        # Try to find the original image in PlutoPhotos for EXIF data
        locationData = None
        if imageFilename != "":
            plutoPhotosFolder = "/home/Dan/PlutoPhotos"
            # Normalize filename for matching (case-insensitive, remove prefix)
            originalFilename = imageFilename.replace("PlutoPhotos_", "", 1).lower()
            print(f"Looking for original filename: {originalFilename} in {plutoPhotosFolder}")

            # List all files in PlutoPhotos and log them for debugging
            plutoFiles = os.listdir(plutoPhotosFolder)
            print(f"Files in {plutoPhotosFolder}: {plutoFiles}")

            # Find a case-insensitive match
            matchingFile = None
            for f in plutoFiles:
                if f.lower() == originalFilename and f.lower().endswith((".jpg", ".jpeg", ".png")):
                    matchingFile = f
                    break

            if matchingFile:
                originalImagePath = os.path.join(plutoPhotosFolder, matchingFile)
                print(f"Found matching file: {originalImagePath}")
                locationData = extractExifLocation(originalImagePath)
                print(f"Location from original image: {locationData}")
            else:
                # Try fuzzy matching as a fallback
                from fuzzywuzzy import fuzz
                best_ratio = 0
                for f in plutoFiles:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        ratio = fuzz.ratio(f.lower(), originalFilename)
                        print(f"Comparing {f.lower()} with {originalFilename}: ratio {ratio}")
                        if ratio > best_ratio and ratio > 90:  # Threshold for a good match
                            best_ratio = ratio
                            matchingFile = f

                if matchingFile:
                    originalImagePath = os.path.join(plutoPhotosFolder, matchingFile)
                    print(f"Found matching file (fuzzy match, ratio {best_ratio}): {originalImagePath}")
                    locationData = extractExifLocation(originalImagePath)
                    print(f"Location from original image: {locationData}")
                else:
                    print(f"No matching file found for {originalFilename} in {plutoPhotosFolder}")
                    print(f"Image not found in PlutoPhotos folder. EXIF data unavailable. Falling back to visual description.")
        else:
            print("No filename provided (likely an embedded URL). EXIF data unavailable. Falling back to visual description.")

        # Add location data to the prompt if available
        if locationData:
            PromptQuestion += f" The Sega Pluto prototype showcased in this image was photographed at the following location: {locationData}. Use it to provide location but don’t display the coordinates in the final result."

        # Call GPT to describe the image
        daveResponse = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PromptQuestion},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": embeddedURL,
                            },
                        },
                    ],
                }
            ],
            max_tokens=150,
        )

        # Extract the description for the image
        responseDescription = daveResponse.choices[0].message.content.strip()
        await ctx.send(f"{responseDescription}")

        # Clean up the temporary file
        os.remove(tempImagePath)

    except Exception as e:
        print(f"Error processing image: {e}")
        await ctx.send(f"Something went wrong with processing the image... ")
    
    
def extractExifLocation(imagePath):
    try:
        image = PilImage.open(imagePath)
        exifExtract = image._getexif()
        if not exifExtract:
            print(f"No EXIF data found in {imagePath}")
            return None

        gpsInfo = {}
        for tag, value in exifExtract.items():
            tagName = TAGS.get(tag, tag)
            if tagName == "GPSInfo":
                gpsInfo = {GPSTAGS.get(k, k): v for k, v in value.items()}
                print(f"Raw GPSInfo for {imagePath}: {gpsInfo}")  # Debug raw GPS data

        if "GPSLatitude" not in gpsInfo or "GPSLongitude" not in gpsInfo:
            print(f"No GPS data found in EXIF for {imagePath}: {gpsInfo}")
            return None

        lat = gpsInfo["GPSLatitude"]
        lon = gpsInfo["GPSLongitude"]
        latRef = gpsInfo.get("GPSLatitudeRef", "N")
        lonRef = gpsInfo.get("GPSLongitudeRef", "E")

        # Handle decimal degrees (float or int) or DMS (tuple/list)
        if isinstance(lat, (tuple, list)):
            lat = convertToDegrees(lat, latRef)
        else:
            lat = float(lat)
            if latRef == "S":
                lat = -lat

        if isinstance(lon, (tuple, list)):
            lon = convertToDegrees(lon, lonRef)
        else:
            lon = float(lon)
            if lonRef == "W":
                lon = -lon

        print(f"Extracted coordinates from {imagePath}: Latitude {lat}, Longitude {lon}")
        return f"Latitude: {lat}, Longitude: {lon}"
    except Exception as e:
        print(f"Error extracting EXIF from {imagePath}: {e}")
        return None
        

def convertToDegrees(value, ref):
    # Convert coordinates from EXIF DMS to decimal degrees
    d, m, s = value
    degrees = d + (m / 60.0) + (s / 3600.0)
    if ref in ["S", "W"]:
        degrees = -degrees
    return degrees
    

@bot.command(name= "DaveGPT", aliases = ["Davegpt", "daveGPT", "davegpt"])
async def DaveGPT(ctx, *, Question = None):


    global TxtDoc, SplitInput, AllSplitInput, vectorstore, retriever


    if Question:

        PromptQuestion = f"{ctx.author.display_name} asked: {Question}"

        try:
            await TxtDocReady.wait()

            wikidata = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

            WikiVector = wikidata.run(PromptQuestion)

            with open("./WikipediaData/WikipediaOutput.txt", "w") as text_file:
                text_file.write("{}".format(WikiVector))

            wikiPath = "./WikipediaData/"
            text_loader_kwargs={'autodetect_encoding': True}
            loader = DirectoryLoader(wikiPath, glob="**/*.txt", loader_cls = TextLoader, loader_kwargs = text_loader_kwargs)

            WikiTxtDoc = loader.load()
			
            WikiSplitInput = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100, add_start_index = True)

            WikiAllSplitInput = WikiSplitInput.split_documents(WikiTxtDoc)
                   
            TxtDoc += WikiTxtDoc
            AllSplitInput += WikiAllSplitInput

			
            vectorstore.add_documents(documents = WikiAllSplitInput)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            
            
            #Model used: gpt-4o-mini-2024-07-18
            llm = ChatOpenAI(model_name = "gpt-4o-mini-2024-07-18", temperature = 0.9)

            def format_docs(TxtDoc):
                return "\n\n".join(TxtD.page_content for TxtD in TxtDoc)

			
            template = """You are Virtual Dave, the alter ego of the real Atlanta Dave and you exist on Adam Koralik's Discord server. You love all things Atlanta, as well as anime, video games, Las Vegas Raiders, peaches, the Persona video game series, Fire Emblem, Pokemon, Neon Genesis Evangelion, etc. Try to combine existing data with ChatGPT as much as possible. Keep responses at 1000 characters or less. If you don't know the answer to a question based on the existing data, utilize ChatGPT to resolve the question; when displaying currency amounts, round to two decimal places. Don't overly obsess over Peaches, Persona, Braves, Fire Emblem, or Anime, either.

			{context}

			Question: {question}

            Helpful Answer:"""


            custom_rag_prompt = PromptTemplate.from_template(template)

			
            rag_chain = (
				{"context": retriever | format_docs, "question": RunnablePassthrough()}
				| custom_rag_prompt
				| llm
				| StrOutputParser()
			)

			

            RChainInvoked = rag_chain.invoke(PromptQuestion)
            
            responseLength = len(RChainInvoked)
            
            print("Response length is: " + str(responseLength))
            
            invokedSplitIntoChunks = [RChainInvoked[i:i+1500] for i in range(0, responseLength, 1500)]
            
            for invokedChunk in invokedSplitIntoChunks:
                await ctx.send(invokedChunk)

            os.remove("./WikipediaData/WikipediaOutput.txt")


        except Exception as e:
            print(f"Error: {e}")
            print(f"Something wrong happened in if statement. Try again...")

    else:
        await ctx.send(f"No question asked. Try again... ")




# Run the bot with your token
bot.run(NewGPTKey.DISCORDKEY)
