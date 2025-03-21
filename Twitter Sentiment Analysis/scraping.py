# Import Dependencies
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from time import sleep
import pandas as pd


# Setup WebDriver with ChromeOptions
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=chrome_options)
driver.get("https://twitter.com/login")


# Twitter credentials (Insert your own)
username_str = ""
password_str = ""


# Setup the login
try:
    username = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[@name='text']"))
    )
    username.send_keys(username_str)
    next_button = driver.find_element(By.XPATH, "//span[contains(text(),'Next')]")
    next_button.click()
except (NoSuchElementException, TimeoutException, WebDriverException) as e:
    print(f"Error during login: {e}")
    driver.quit()

try:
    password = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[@name='password']"))
    )
    password.send_keys(password_str)
    log_in = driver.find_element(By.XPATH, "//span[contains(text(),'Log in')]")
    log_in.click()
except (NoSuchElementException, TimeoutException, WebDriverException) as e:
    print(f"Error during login: {e}")
    driver.quit()

# Search for the hashtag #BlackLivesMatter
try:
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//input[@data-testid='SearchBox_Search_Input']"))
    )
    search_box.send_keys("Black Lives Matter")
    search_box.send_keys(Keys.ENTER)
except (NoSuchElementException, TimeoutException, WebDriverException) as e:
    print(f"Error during search: {e}")
    driver.quit()

# Click on the "People" tab
try:
    people_tab = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//span[contains(text(),'People')]"))
    )
    people_tab.click()
except (NoSuchElementException, TimeoutException, WebDriverException) as e:
    print(f"Error during switching to people: {e}")
    driver.quit()

# Click on the first profile in the "People" tab
try:
    first_profile = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//*[@id='react-root']/div/div/div[2]/main/div/div/div/div/div/div[3]/section/div/div/div[1]/div/div/button/div/div[2]/div[1]/div[1]/div/div[1]/a/div/div[1]/span/span[1]"))
    )
    first_profile.click()
except (NoSuchElementException, TimeoutException, WebDriverException) as e:
    print(f"Error during profile selection: {e}")
    driver.quit()

# Initialize lists to store tweet data
UserTags = []
TimeStamps = []
Tweets = []
Replies = []
Retweets = []
Likes = []

# Fetch tweets from the profile
try:
    sleep(10)
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        articles = driver.find_elements(By.XPATH, "//article[@data-testid='tweet']")
        for article in articles:
            try:
                UserTag = article.find_element(By.XPATH, ".//div[@data-testid='User-Name']").text
                TimeStamp = article.find_element(By.XPATH, ".//time").get_attribute('datetime')
                Tweet = article.find_element(By.XPATH, ".//div[@data-testid='tweetText']").text
                Reply = article.find_element(By.XPATH, ".//button[@data-testid='reply']").text
                Retweet = article.find_element(By.XPATH, ".//button[@data-testid='retweet']").text
                Like = article.find_element(By.XPATH, ".//button[@data-testid='like']").text

                UserTags.append(UserTag)
                TimeStamps.append(TimeStamp)
                Tweets.append(Tweet)
                Replies.append(Reply)
                Retweets.append(Retweet)
                Likes.append(Like)
            except NoSuchElementException as e:
                print(f"Error: {e}")

        driver.execute_script('window.scrollTo(0,document.body.scrollHeight);')
        sleep(5)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:  # Break if no new tweets are loaded
            break
        last_height = new_height

        # Optionally break after a certain number of tweets
        if len(Tweets) > 4000:  # Adjust as needed
            break

    # Print the number of collected tweets
    print(len(UserTags), len(TimeStamps), len(Tweets), len(Replies), len(Retweets), len(Likes))

    # Save data to a DataFrame and CSV file
    df = pd.DataFrame(zip(UserTags, TimeStamps, Tweets, Replies, Retweets, Likes),
                      columns=['UserTags', 'TimeStamps', 'Tweets', 'Replies', 'Retweets', 'Likes'])

    df.head()

    # Save the DataFrame to a CSV file in the current working directory
    df.to_csv("tweets_live.csv", index=False)

    print("Tweets have been saved to 'tweets_live.csv'")

except WebDriverException as e:
    print(f"Error during tweet fetching: {e}")
finally:
    # Close the driver
    driver.quit()