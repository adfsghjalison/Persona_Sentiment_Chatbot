# Sentiment-Dialogue
add sentiment to dialogue seq2seq decoder at each time step  
## How to run :
training:  
Go to personal-dialogue directory, then run:  
* `$ python3 main.py -train`

testing:  
Go to personal-dialogue directory, then run:  
get model: 

* `$ wget https://www.csie.ntu.edu.tw/~b02902076/sentiment_model.tar`  
* `$ tar -zxvf sentiment_model.tar`  
Run stdin test:  
* `$ python3 main.py -stdin`

## example:
```
1.0:i love you
response:love is all the song you can do and you can visit
```
```
0.95:i love you
response:i ' m a good man .
```
```
0.4:i love you  
response:you ' re not the one i want
```
```
0.1:i love you  
response:i wish i could do that
```
```
1.0:how are you
response:i am good
```
```
0.0:how are you
response:i ' m so embarrassed .
```
```
0.95:i want to  leave
response:i ' il go with you .
```
```
0.0:i want to leave
response:i wish i could do that with you .
```
## environment:
tensorflow-gpu1.0.0 or 1.1.0

