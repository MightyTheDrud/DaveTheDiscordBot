Dave the Discord Bot

A versatile Discord bot designed to provide a wide range of functionalities from simple commands to advanced integrations with external APIs and services. This bot is optimized for performance on low-power devices like Raspberry Pi or older laptops.

Features

**General Commands:**
  - `!Commands`: Lists available commands.
  - `!Davisms`: Randomly displays pinned messages from the channel.
  - `!AskDave`: Provides random responses or answers to user questions.
  - `!Koralik`: Shares random Adam Koralik videos.
  - `!JimmyCarter`: Displays trivia about Jimmy Carter.

**Financial Tools:**
  - `!Lodmot` or `!Stocks`: Fetches current stock prices and changes for various symbols.
  - `!Price`: Looks up prices of items or games on eBay, including region-specific searches.

**Search and Information:**
  - `!Image`: Searches and displays images from Google Images.
  - `!Christory`: Retrieves "Today in Christory" from the CWCki.
  - `!Dictionary`: Shows random definitions from Urban Dictionary.
  - `!What`: Repeats the last non-bot message in all caps.

**Weather and Conversion:**
  - `!Weather`: Fetches weather information using OpenWeatherMap API.
  - **Temperature Conversion:** Automatically converts temperatures mentioned in messages to Fahrenheit, Celsius, and Kelvin.

**Photo Sharing:**
  - `!Pluto` or `!Sega`: Shares random photos from the PlutoPhotos folder.
  - `!AtGames`: Shares random photos from the AtGamesPhotos folder.

**AI Integration:**
  - `!DaveGPT`: Uses AI to answer questions based on local text files and Wikipedia queries.
  - `!Describe`: Describes images replied to in the chat with AI assistance.

Installation - Install the following packages on through a virtual environment to avoid conflicts
- discord.py
- openai
- requests
- BeautifulSoup4
- numpy
- yfinance
- Pillow
- forex-python
- langchain with associated community, chroma, and openai packages
- jsons

Usage
- Prefix all commands with !
- For commands that require parameters, follow the command with the parameter, e.g., !Price item Nintendo Switch.

Notes
- The bot uses external APIs which might have rate limits or require API keys.
- Ensure all media files like images for !Pluto and !AtGames are properly placed in their designated directories.




