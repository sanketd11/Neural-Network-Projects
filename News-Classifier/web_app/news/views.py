from django.shortcuts import render
from django.http import HttpResponse
import json, json
from .classify import Predictor
predictor = Predictor()

def index(request):
    return render(request,'news/index.html')


def result(request):
    if request.method == 'POST':
        text = request.POST['news_textbox']
        output = predictor.predict(text)

    return render(request,'news/result.html', context={'category':output, 'news': text})


