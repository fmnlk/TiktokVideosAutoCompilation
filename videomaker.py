#!/usr/bin/python3

###############
### Modules ###
###############

import os
import json
import asyncio
import argparse
import datetime
import pandas as pd
from datetime import date
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from moviepy import VideoFileClip, concatenate_videoclips

TRENDING_DB = 'dataVideo.txt'
CHALLENGE_DB = 'dataVideoChallenge.txt'
CHALLENGE_LIST = 'listChallenge.txt'

STAT_COLUMNS = ['likeCount', 'shareCount', 'playCount', 'commentCount']
#a video is kept in the DB if at least one of its stats is above the threshold
THRESHOLDS = {'likeCount': 300000, 'shareCount': 5000, 'playCount': 2000000, 'commentCount': 3000}

#############
### DB IO ###
#############

def loadDB(file, index='id'):
    """
    Load a json DB file into a DF indexed by the given column
    """
    with open(file, 'r') as f:
        videos_dict = json.load(f)
    DB = pd.DataFrame.from_dict(videos_dict)
    DB[index] = DB[index].astype(str)
    DB.set_index(index, inplace=True)
    return DB

def saveDB(DB, file, index='id'):
    """
    Save a DF indexed by the given column into a json DB file
    """
    DB = DB.copy()
    #putting back the index as a column to have it in the export
    DB[index] = DB.index
    DB.to_json(file, orient="records")

def upsertDB(DB, newData, keepColumns):
    """
    Insert the rows of newData that are not in DB, and update the other ones
    without touching the columns in keepColumns (ex: videoUsed)
    """
    numOldRecord = len(DB)
    #adding all the data that are not in DB = insert
    DB = pd.concat([DB, newData[~newData.index.isin(DB.index)]])
    #updating the data = updating only the numbers
    DB.update(newData.drop(columns=keepColumns, errors='ignore'))
    numNewRecord = len(DB)
    print("Number of records added in DB:", numNewRecord - numOldRecord)
    print("Total number of records:", numNewRecord)
    return DB

def isPopular(dic):
    return any(dic[stat] > threshold for stat, threshold in THRESHOLDS.items())

#######################
### Data collection ###
#######################

def videoToDict(video, musicId=None):
    """
    Extract the info we want to save from a TikTokApi Video
    """
    data = video.as_dict
    stats = video.stats or {}
    dic = {}
    dic['id'] = str(video.id)
    if musicId is not None:
        dic['musicId'] = str(musicId)
    dic['timeCreated'] = int(data.get('createTime', 0))
    dic['likeCount'] = int(stats.get('diggCount', 0))
    dic['shareCount'] = int(stats.get('shareCount', 0))
    dic['playCount'] = int(stats.get('playCount', 0))
    dic['commentCount'] = int(stats.get('commentCount', 0))
    #author is needed to build the video url for the download
    author = data.get('author') or ''
    dic['author'] = author.get('uniqueId', '') if isinstance(author, dict) else str(author)
    dic['videoUsed'] = False
    dic['videoUsedDate'] = ''
    return dic

async def collectVideos(sources, count):
    """
    Open a TikTok session and collect the videos of every source
    INPUT: list of (musicId or None for trending), number of videos per source
    OUTPUT: list of dictionnary with processed video data
    """
    #imported here so the video making part doesn't need playwright
    from TikTokApi import TikTokApi

    #ms_token is the msToken cookie of a tiktok.com browser session, it greatly reduces bot detection
    msToken = os.environ.get('MS_TOKEN')
    listOfVideoDic = []
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[msToken] if msToken else None, num_sessions=1, sleep_after=3,
                                  browser=os.environ.get('TIKTOK_BROWSER', 'chromium'))
        for musicId in sources:
            if musicId is None:
                print("Getting trending video data...")
                videos = api.trending.videos(count=count)
            else:
                print("Getting the challenge data for music", musicId)
                videos = api.sound(id=musicId).videos(count=count)
            try:
                async for video in videos:
                    listOfVideoDic.append(videoToDict(video, musicId))
            except Exception as e:
                print(e)
                print("Error to get the video data, keeping what was retrieved")
    return listOfVideoDic

def toDF(listOfVideoDic):
    newDataDF = pd.DataFrame(listOfVideoDic)
    print("Number of row:", newDataDF.shape[0])
    if newDataDF.empty:
        return newDataDF
    newDataDF.drop_duplicates(subset='id', inplace=True, keep='last')
    print("Number of row without duplicates:", newDataDF.shape[0])
    newDataDF.set_index('id', inplace=True)
    return newDataDF

def importTrendingDataToDB(count):
    """
    Update the DB with new trending video
    """
    listOfVideoDic = [dic for dic in asyncio.run(collectVideos([None], count)) if isPopular(dic)]
    newDataDF = toDF(listOfVideoDic)
    if newDataDF.empty:
        print("No new trending video")
        return
    print("merging data into DB")
    DB = upsertDB(loadDB(TRENDING_DB), newDataDF, ['videoUsed', 'videoUsedDate'])
    saveDB(DB, TRENDING_DB)

def importChallengeDataToDB(count, musicIds=None):
    """
    Update the challenge DB with the videos of the given musics (default: every music of listChallenge.txt not used yet)
    """
    if not musicIds:
        challengeDB = loadDB(CHALLENGE_LIST, index='musicId')
        musicIds = challengeDB[challengeDB['challengeUsed'] != True].index.tolist()
    newDataDF = toDF(asyncio.run(collectVideos(musicIds, count)))
    if newDataDF.empty:
        print("No new challenge video")
        return
    DB = upsertDB(loadDB(CHALLENGE_DB), newDataDF, ['videoUsed', 'videoUsedDate'])
    saveDB(DB, CHALLENGE_DB)

def generateLinkFromId(videoId, author=''):
    """
    function to generate a link to a video from its ID (the author is optional)
    """
    return 'https://www.tiktok.com/@' + (author if isinstance(author, str) else '') + '/video/' + str(videoId)

def updateDB(file=TRENDING_DB, prune=False):
    """
    function to update the data about all the video in the DB
    1 request per video so the DB has to stay small
    INPUT: prune = remove the videos that can't be retrieved anymore
    """
    DB = loadDB(file)
    videoData = []
    failed = []
    with YoutubeDL({'quiet': True, 'skip_download': True}) as ydl:
        for videoId in DB.index:
            print(videoId)
            try:
                info = ydl.extract_info(generateLinkFromId(videoId, DB.at[videoId, 'author'] if 'author' in DB else ''), download=False)
            except DownloadError as e:
                print(e)
                failed.append(videoId)
                continue
            dic = {'id': videoId}
            dic['commentCount'] = info.get('comment_count')
            dic['likeCount'] = info.get('like_count')
            dic['playCount'] = info.get('view_count')
            dic['shareCount'] = info.get('repost_count')
            print(dic)
            videoData.append(dic)

    if videoData:
        DB.update(pd.DataFrame(videoData).set_index('id'))
    print("Number of videos that couldn't be retrieved:", len(failed))
    if prune and failed:
        print("deleting them from the DB")
        DB.drop(failed, inplace=True)
    saveDB(DB, file)

def filterTrendingVideo(file=TRENDING_DB):
    """
    Remove from the DB the videos that are not popular enough
    """
    DB = loadDB(file)
    print(len(DB))
    filtered_df = DB[DB.apply(isPopular, axis=1)]
    print(len(filtered_df))
    saveDB(filtered_df, file)

####################
### Video making ###
####################

def loadDbIntoDf2(content, musicId=None):
    """
        Function that load the json file into a DF
        INPUT: content: trending/music, musicId to filter on for music
        OUTPUT; dataframes and columns
    """
    file = CHALLENGE_DB if content == 'music' else TRENDING_DB
    with open(file, 'r') as f:
        videos_dict = json.load(f)
    df = pd.DataFrame.from_dict(videos_dict)
    df['id'] = df['id'].astype(str)
    #filter on challenge
    if content == 'music' and musicId is not None:
        df = df[df.musicId.astype(str) == str(musicId)]
    return df

def selectTop(dfProcess, period, periodNumber, ranking, year=None, number=50, skipUsed=False):
    """
        Function to select the top videos for a given period and a given scoring system
        INPUT: DataFrame, period: month/week/all, ranking: trending/share/like/view
        OUTPUT: New DataFrame  with only x top videos sorted by score

    """
    dfProcess = dfProcess.copy()
    if skipUsed:
        dfProcess = dfProcess[dfProcess['videoUsed'] != True]
    #creating new columns
    dfProcess['timeCreated'] = pd.to_datetime(pd.to_numeric(dfProcess['timeCreated']), unit='s')
    if year is not None:
        dfProcess = dfProcess[dfProcess['timeCreated'].dt.year == year]
    if period == 'week':
        dfProcess = dfProcess[dfProcess['timeCreated'].dt.isocalendar().week == periodNumber]
    elif period == 'month':
        dfProcess = dfProcess[dfProcess['timeCreated'].dt.month == periodNumber]
    elif period != 'all':
        print("Period parameter is unknown")
    #select useful columns
    columns_name = ['id', 'author', 'commentCount', 'likeCount', 'playCount', 'shareCount']
    dfProcess = dfProcess.reindex(columns=columns_name)
    #calculating score
    normalized = dfProcess[STAT_COLUMNS].astype(float)
    normalized = normalized / normalized.max() #normalisation
    if ranking == 'trending':
        score = (35/100 * normalized['likeCount'] + 20/100*normalized['playCount'] + 35/100* normalized['shareCount']
    + 10/100*normalized['commentCount'])*100
    elif ranking == 'share':
        score = normalized['shareCount']
    elif ranking == 'like':
        score = normalized['likeCount']
    elif ranking == 'view':
        score = normalized['playCount']
    else:
        print('ranking unknown')
        score = normalized['playCount']
    dfProcess['score'] = score
    #selecting top
    dfProcess = dfProcess.sort_values('score', ascending=False)
    return dfProcess.head(number)

def download(df_shorter, folderName):
    """
        Functions to download videos given their ID.
        INPUT: DataFrame
        OUTPUT: list of (id, path) of the videos dowloaded and stored in the folder
    """
    os.makedirs(str(folderName), exist_ok=True)
    vid_dl = []
    i = 1
    for videoId, author in zip(df_shorter['id'], df_shorter['author']):
        path = os.path.join(str(folderName), str(i) + '.mp4')
        options = {'outtmpl': path, 'format': 'best[ext=mp4]/best', 'quiet': True, 'overwrites': True}
        try:
            with YoutubeDL(options) as ydl:
                ydl.download([generateLinkFromId(videoId, author)])
        except DownloadError as e:
            print(e)
            print("Error to download video", videoId)
            continue
        vid_dl.append((videoId, path))
        i = i + 1
    return vid_dl

def merge(vidlist, outputName):
    """
        Function to merge videos dowloaded in one video.
        INPUT: list of videos downloaded
        OUTPUT: One video saved as outputName.mp4
    """
    #resizing video
    clips = []
    for vid in vidlist:
        if vid.endswith(".mp4"):
            video = VideoFileClip(vid)
            ratio = video.h / video.w
            if ratio < (16/9 - 0.01):
                video = video.resized(width=1080)
            else:
                video = video.resized(height=1920)
            clips.append(video)
    if not clips:
        print("No video to merge")
        return None
    finalrender = concatenate_videoclips(clips, method='compose')
    output = str(outputName) + '.mp4'
    finalrender.write_videofile(output, codec='libx264')
    for clip in clips:
        clip.close()
    return output

def update(df, usedIds, file=TRENDING_DB):
    """
        Function to update videoUsed and videoUsedDate info in the original dataframe
        and save it as json.
    """
    d = date.today().strftime("%Y_%m_%d")
    used = df['id'].isin(usedIds)
    df.loc[used, 'videoUsed'] = True
    df.loc[used, 'videoUsedDate'] = d
    df.to_json(file, orient="records")

def makeVideo(content, period, periodNumber, ranking, year=None, number=50, musicId=None, skipUsed=False):
    """
    function that call the different function to make a video
    """
    df = loadDbIntoDf2(content, musicId)
    df_shorter = selectTop(df, period, periodNumber, ranking, year, number, skipUsed)
    if df_shorter.empty:
        print("No video for this period")
        return None
    outputName = '_'.join(str(x) for x in [content, year, period, periodNumber] if x is not None)
    vid_dl = download(df_shorter, outputName)
    output = merge([path for _, path in vid_dl], outputName)
    if output is not None:
        #marking the videos as used in the full DB (not the filtered one)
        update(loadDbIntoDf2(content), [videoId for videoId, _ in vid_dl], CHALLENGE_DB if content == 'music' else TRENDING_DB)
        print("Video saved in", output)
    return output

###########
### CLI ###
###########

def main():
    parser = argparse.ArgumentParser(description="Automatic compilation of the best TikTok videos")
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('import-trending', help="add the current trending videos to " + TRENDING_DB)
    p.add_argument('--count', type=int, default=300, help="number of trending videos to retrieve")

    p = sub.add_parser('import-music', help="add the videos of music challenges to " + CHALLENGE_DB)
    p.add_argument('--count', type=int, default=100, help="number of videos to retrieve per music")
    p.add_argument('--music-id', nargs='*', help="music IDs (default: the unused ones of " + CHALLENGE_LIST + ")")

    p = sub.add_parser('update-db', help="refresh the stats of every video of a DB")
    p.add_argument('--content', choices=['trending', 'music'], default='trending')
    p.add_argument('--prune', action='store_true', help="delete the videos that can't be retrieved anymore")

    p = sub.add_parser('filter', help="remove the videos that are not popular enough from " + TRENDING_DB)

    p = sub.add_parser('make-video', help="download the top videos of a period and merge them")
    p.add_argument('--content', choices=['trending', 'music'], default='trending')
    p.add_argument('--music-id', help="music ID to filter on when content is music")
    p.add_argument('--period', choices=['week', 'month', 'all'], default='week')
    p.add_argument('--period-number', type=int, help="week or month number (default: the current one)")
    p.add_argument('--year', type=int, help="only keep videos created this year")
    p.add_argument('--ranking', choices=['trending', 'share', 'like', 'view'], default='trending')
    p.add_argument('--number', type=int, default=50, help="number of videos in the compilation")
    p.add_argument('--skip-used', action='store_true', help="ignore videos already used in a compilation")

    args = parser.parse_args()
    if args.command == 'import-trending':
        importTrendingDataToDB(args.count)
    elif args.command == 'import-music':
        importChallengeDataToDB(args.count, args.music_id)
    elif args.command == 'update-db':
        updateDB(CHALLENGE_DB if args.content == 'music' else TRENDING_DB, args.prune)
    elif args.command == 'filter':
        filterTrendingVideo()
    elif args.command == 'make-video':
        periodNumber = args.period_number
        if periodNumber is None and args.period != 'all':
            today = datetime.date.today()
            periodNumber = today.isocalendar()[1] if args.period == 'week' else today.month
        makeVideo(args.content, args.period, periodNumber, args.ranking, args.year, args.number, args.music_id, args.skip_used)

if __name__ == '__main__':
    main()
