import whisper_timestamped as whisper
from nltk.corpus import stopwords
from nltk import download


def speech_analysis(file_path):
    audio = whisper.load_audio(file_path)
    model = whisper.load_model("medium", device="cpu")
    result = whisper.transcribe(model, audio, language="ru", detect_disfluencies=True)
    download('stopwords')

    output = {}
    stop_words = stopwords.words("russian")
    data = result['segments']
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
    most_frequent = sorted(output.items(), key=lambda x: x[1], reverse=True)[:3]
    print("Самые чистотные слова:", )
    for i in most_frequent:
        if i[0] == "":
            continue
        print(f"{i[0]}: {i[1]}")
    print("Процент эээээ к общей речи:", ammm_precent)

    return output
