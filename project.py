#Director
# Importing
import pandas as pd
df = pd.read_csv('tmdb_5000_credits.csv')

# Adding director column
import json
director_list = []
for i in range(df.shape[0]):
    crew_list = json.loads(df.iloc[i, 3])
    for crew_dict in crew_list:
        if crew_dict['job'] == 'Director':
            director = crew_dict['name']
        else:
            continue
    director_list.append(director)
df['director'] = director_list

# Scraping director rank
#function of getting 25 directors
def get_25_direct(url):
    import requests
    import re
    from bs4 import BeautifulSoup
    #getting data
    try:
        response = requests.get(url)
        if not response.status_code == 200:
            print('wrong status')
        try:
            content = BeautifulSoup(response.content, 'lxml')
        except:
            print('wrong while parsing')
    except:
        print('wrong while requesting')
    # Getting rank & name    
    rank_section = content.find_all('li', {'class': re.compile('gridItem_main__1ilxA gridItem_hasMedia__38WR2*')})
    rank_list = []
    for section in rank_section:
        rank = section.find('strong', {'class':'gridItem_rank__3Q_TO'}).get_text()
        name = section.find('div', {'class': 'gridItem_nameWrapper__2KPQS'}).find('h2').get_text()
        rank_list.append((rank, name))
    return rank_list 

#get all ranks
rank_list = []
for i in range(23):
    url = 'https://www.ranker.com/crowdranked-list/the-most-oscar-worthy-directors-of-all-time?page=' + str(i+1)
    rank_list.extend(get_25_direct(url))
director_rank_df = pd.DataFrame(rank_list, columns = ['rank', 'director_name'])

# matching ranks
df_joined = df.merge(director_rank_df, how = 'left', right_on = 'director_name', left_on = 'director')

#cleaning
import numpy as np
df_joined.drop('director_name', axis = 1, inplace = True)
df_joined['rank'].fillna('556', inplace = True)
df_joined.head()
df_director = df_joined
df_director.head()


#Actors
import requests
import re
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import json
def get_actor_rank():
    actor_list = []
    for page_number in range(1,11):    
        url = f"https://www.imdb.com/list/ls058011111/?sort=list_order,asc&mode=detail&page={page_number}"
        response = requests.get(url)
        if not response.status_code == 200:
            None
        try:
            results_page = BeautifulSoup(response.content,'lxml') 
        except:
            None
        results_section = results_page.find_all("div",{"class":"lister-item mode-detail"})
        for result in results_section:
            rank = result.find("span",{"class":"lister-item-index unbold text-primary"}).get_text().replace(".","").strip()
            name = result.find("h3",{"class":"lister-item-header"}).find("a").get_text().strip()
            actor_list.append((rank,name))
    return actor_list
# n means top n actors in a movie
def get_top_n_starring(n):
    df0 = pd.read_csv("tmdb_5000_credits.csv",na_values="null")
    movies_actors_data = df0["cast"].tolist()
    starring_list = []
    for movie_actors_data in movies_actors_data:
        movie_actors_data = json.loads(movie_actors_data)
        actors = []
        for movie_actor_data in movie_actors_data:
            name = movie_actor_data["name"]
            actors.append(name)
            main_actors = actors[0:n]
        starring_list.append(main_actors)
    return starring_list
# n means top n actors in a movie
def get_actorRank(n):
    
    #generate a dataframe with rank and actor's name
    
    rank_chart = pd.DataFrame(get_actor_rank())
    rank_chart = rank_chart.rename(columns={0:"rank",1:"name"})
    rank_chart.set_index("rank")

    
    #generate a dataframe of top n starring
    
    df2 = pd.DataFrame(get_top_n_starring(n))
    for i in range(n):
        df2 = df2.rename(columns={i:f"actor{i+1}"})
    df2["movies_name"] = pd.read_csv("tmdb_5000_credits.csv",na_values="null")["title"]
    #use a column to control the order
    order_keeper = []
    for i in range(4803):
        order_keeper.append(i)
    df2["order_keeper"] = order_keeper

    #merge the dataframe in the way of vlookup(in excel)
    
    merge_results = df2
    for i in range(n):        
        merge_results = pd.merge(merge_results,rank_chart,left_on = f"actor{i+1}",right_on = "name",how='left')
        merge_results = merge_results.rename(columns={"rank": f"rank{i+1}"})
        merge_results = merge_results.drop(['name'], axis=1)
        merge_results[f"rank{i+1}"].fillna(1000,inplace=True)
    merge_results = merge_results.sort_values(by="order_keeper",ascending = True)
    return merge_results
df_actor = get_actorRank(3)

#Task 1
datafile1 = "/Users/jerryyileibao/Downloads/archive/tmdb_5000_movies.csv"
datafile2 = "/Users/jerryyileibao/Downloads/CPI_1916-2020.csv"
df1 = pd.read_csv(datafile1)
df2 = pd.read_csv(datafile2)
#change the release_date type into datetime64
df1['release_date'] = pd.to_datetime(df1['release_date'],format = '%Y-%m-%d')
#check the earliest release date
df1['release_date'].min()
#extract year from the release date; type of year is float64
df1['year']=pd.DatetimeIndex(df1['release_date']).year
#check to see if there is any null value in the year column
df1['year'].isnull().sum()
#for the null year value, fill in 0
df1['year']=df1['year'].fillna(0)
#check to see if there is any null value in the year column
#there are none!
df1['year'].isnull().sum()
#convert year column from float to int
df1['year']=df1['year'].astype(int)
df2['year'] = df2['year'].astype(int)
#drop 
df2=df2.drop('Unnamed: 2',1)
#merge df2 (CPI) with df1 (movie) to add inflation data to df1
df1=df1.merge(df2,on='year')
#calculate the real budget based on CPI data
df1['real_budget']=df1['budget']/(df1['CPI']/100)
df1['real_revenue']=df1['revenue']/(df1['CPI']/100)
df_task_1 = df1


#Holidays
import pandas as pd
import numpy as np
# !pip install holidays
import datetime
from datetime import date
import holidays
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
def fix_data(datefile):
    df = pd.read_csv(datefile)
    df = df[df['release_date'].notnull()]
    # fix dates
    df['Release_date'] = df['release_date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    df['Release_month'] = df['Release_date'].apply(lambda x: x.month)
    df['Release_year'] = df['Release_date'].apply(lambda x: x.year)
    
    us_holidays = holidays.US()
    df['US_holidays'] = np.nan
    for index in df.index:
        if df.loc[index, 'Release_date'] in us_holidays:
            df.loc[index, 'US_holidays'] = 'True'
        else:
            df.loc[index, 'US_holidays'] = 'False'
    
    all_dates = df['US_holidays'].values
    encoder = LabelEncoder()
    encoder.fit(all_dates)
    coded_array = encoder.transform(all_dates)
    k = len(coded_array)
    n_labels = len(np.unique(coded_array))
    one_hot = np.zeros((k,n_labels))
    one_hot[np.arange(k), coded_array] = 1
    dummy = pd.DataFrame(one_hot, columns = ['not_holidy', 'holiday'])
    df = pd.concat([df, dummy], axis=1, join='inner')
    return df
datafile = "tmdb_5000_movies.csv"
dataframe = fix_data(datafile)
df_holiday = dataframe


#Task 5
import pandas as pd
import numpy as np
import json
from ast import literal_eval
#dealing with production_countries column
df = pd.read_csv("tmdb_5000_movies.csv")
df['production_countries'].iloc[2]
#add a column which pulls out the specific countries name of production_countries
production_countries = []
for i in range(df.shape[0]):
    countries_list = json.loads(df['production_countries'].iloc[i])
    countries = []
    for countries_dict in countries_list:
        country = countries_dict['iso_3166_1']
        countries.append(country)
    production_countries.append(countries)
df['production_countries_name'] = production_countries
#create a column which only has value 0 and 1(1 means production_countries has US)
US_or_not = list()
for i in range(df.shape[0]):
    if 'US' in df['production_countries_name'].iloc[i]:
        US_or_not.append(1)
    else:
        US_or_not.append(0)
df['production_countries_US'] = US_or_not
df_country = df

#Task 3
#define a function that returns a list of all genre names of a movie
import re

def get_genre(genre_list):
    #use regular expression to find all "name"
    genres = re.findall(r'name": "([a-zA-Z]+)',genre_list)
    return genres
#apply the get_genre function to each row to create a list of genre names
df1['genre_names']=df1['genres'].apply(get_genre)
#add all genres of all movies into one tuple to see how many different genres are there
#reference: https://stackoverflow.com/questions/49621540/finding-a-word-after-a-specific-word-in-python-using-regex-from-text-file/49621730
all_genre = list()

for movie in range(len(df1)):
    #for each movie, extend the genres into the all_genre list
    all_genre.extend(df1.loc[movie,'genre_names'])

all_genre = set(all_genre)
print("the number of different genres is: "+str(len(all_genre)))


#Task 7
#similar to how we dealt with genres
#reference: https://stackoverflow.com/questions/49621540/finding-a-word-after-a-specific-word-in-python-using-regex-from-text-file/49621730
def get_language(lan_list):
    #use regex to find all languages spoken
    languages = re.findall(r'name": "([a-zA-Z]+)',lan_list)
    return languages
#apply the get_language function to each row to create a list of languages spoken
df1['language_names']=df1['spoken_languages'].apply(get_genre)
#define a function that returns 1 if there is more than 1 language
def more_lang(lan_list):
    #if no language info for the movie, then return -1
    if len(lan_list)==0:
        return -1
    #if more than 1 langauge, return 1
    if len(lan_list)>1:
        return 1
    #if only one language, return 0
    else:
        return 0
df1['I_other_languages'] = df1['language_names'].apply(more_lang)
df_task_7 = df1
