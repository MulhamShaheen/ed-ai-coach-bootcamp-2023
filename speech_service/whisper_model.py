import whisper_timestamped as whisper
from nltk.corpus import stopwords
from nltk import download
import re

WORDS_PARASITES = [
    "как бы",
    "так вот",
    "короче",
    "типа",
    "принципе",
    "ну",
    "значит",
    "понимаешь",
    "допустим",
    "фактически",
    "всё такое",
    "целом",
    "то есть",
    "это",
    "это самое",
    "как сказать",
    "видишь",
    "слышишь",
    "так сказать",
    "вот",
]


def init_whisper(size="tiny"):
    model = whisper.load_model(size, device="cpu")
    return model


def speech_analysis(file_path):
    audio = whisper.load_audio(file_path)
    model = whisper.load_model("medium", device="cpu", in_memory=True)
    result = whisper.transcribe(model, audio, language="ru", detect_disfluencies=True)
    download('stopwords')

    output = {}
    stop_words = stopwords.words("russian")
    data = result['segments']
    text = result['text']

    parasite_analysis = {}

    for word in WORDS_PARASITES:
        parasite_analysis[word] = text.lower().count(word)

    for seg in data:
        for word in seg['words']:
            text = word['text']
            text = ''.join(x for x in text if x.isalpha())
            if text in stop_words:
                continue
            text = text.lower()
            if text in output.keys():
                output[text] += 1
            else:
                output[text] = 1

    if '' in output.keys():
        ammm_precent = output[''] / sum(output.values())
    else:
        ammm_precent = 0
    most_frequent = sorted(output.items(), key=lambda x: x[1], reverse=True)[:6]
    most_frequent_parasites = sorted(parasite_analysis.items(), key=lambda x: x[1], reverse=True)[:6]
    print("Самые чистотные слова:", )
    for i in most_frequent:
        if i[0] == "":
            continue
        print(f"{i[0]}: {i[1]}")

    for i in most_frequent:
        if i[0] == "":
            continue
        print(f"{i[0]}: {i[1]}")
    print("Процент эээээ к общей речи:", ammm_precent)

    response = {}
    response["words"] = output
    response["parasite"] = parasite_analysis
    response["top words"] = most_frequent
    response["top parasites"] = most_frequent_parasites
    response["overall count"] = sum(output.values())
    response["ammm count"] = output['']

    return response
