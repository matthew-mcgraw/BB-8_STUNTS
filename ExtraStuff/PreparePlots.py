import matplotlib.pyplot as plt
import pandas as pd

data_files_path = 'C:\\Users\\mcgra\\OneDrive\\Documents\\CourseMaterials\\DMU\\FinalProject\\BB-8_Stunts\\RecordedResults\\'





def plot_GoFast_scores():
    data = pd.read_csv(data_files_path+'BB-8_GoFast_EpisodeScore.csv')

    episodes = data['Step']
    scores = data['Value']

    score_smoothed = scores.rolling(window=75).mean()



    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linewidth'] = 0.5

    plt.plot(episodes,score_smoothed)
    plt.xlabel('Episode #',fontsize=14)
    plt.ylabel('Score',fontsize=14)
    plt.title('BB-8 Move Quickly Performance')
    plt.gca().yaxis.grid(True,color='white',linewidth=1)
    plt.gca().xaxis.grid(False)
    plt.gca().tick_params(axis='both', colors='black')
    plt.show()


def plot_GoFast_ActionLoss():
    data = pd.read_csv(data_files_path+'BB-8_GoFast_ActionLoss.csv')
    print(data)
    episodes = data['Step']
    scores = data['Value']

    score_smoothed = scores.rolling(window=100).mean()



    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linewidth'] = 0.5

    plt.plot(episodes,score_smoothed)
    plt.xlabel('Step #',fontsize=14)
    plt.ylabel('Action Loss',fontsize=14)
    plt.title('BB-8 Move Quickly Action Loss')
    plt.gca().yaxis.grid(True,color='white',linewidth=1)
    plt.gca().xaxis.grid(False)
    plt.gca().tick_params(axis='both', colors='black')
    plt.show()

def plot_GOA_scores():
    data = pd.read_csv(data_files_path+'Success_GAO-Simple_06_EpisodeScores.csv')

    episodes = data['Step']
    scores = data['Value']

    score_smoothed = scores.rolling(window=40).mean()



    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linewidth'] = 0.5

    plt.plot(episodes,score_smoothed)
    plt.xlabel('Episode #',fontsize=14)
    plt.ylabel('Score',fontsize=14)
    plt.title('BB-8 Go Around Obstacle Performance')
    plt.gca().yaxis.grid(True,color='white',linewidth=1)
    plt.gca().xaxis.grid(False)
    plt.gca().tick_params(axis='both', colors='black')
    plt.show()


def plot_JTH_scores():
    data = pd.read_csv(data_files_path+'BB-8_JTH_Success_01_EpisodeScores.csv')

    episodes = data['Step']
    scores = data['Value']

    score_smoothed = scores.rolling(window=40).mean()



    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.linewidth'] = 0.5

    plt.plot(episodes,score_smoothed)
    plt.xlabel('Episode #',fontsize=14)
    plt.ylabel('Score',fontsize=14)
    plt.title('BB-8 Jump Through Hoop')
    plt.gca().yaxis.grid(True,color='white',linewidth=1)
    plt.gca().xaxis.grid(False)
    plt.gca().tick_params(axis='both', colors='black')
    plt.show()


#plot_GoFast_scores()
#plot_GoFast_ActionLoss()
#plot_GOA_scores()

plot_JTH_scores()


