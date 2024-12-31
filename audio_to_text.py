from vosk import Model, KaldiRecognizer
import pyaudio
import wave
"""
    Обязательно установить в проект модель Vosk
    Ее можно скачать по ссылке - https://github.com/alphacep/vosk-space/blob/master/models.md
"""

# Укажите путь к вашей модели
CHUNK = 1024 # определяет форму ауди сигнала
FRT = pyaudio.paInt16 # шестнадцатибитный формат задает значение амплитуды
CHAN = 1 # канал записи звука
RT = 44100 # частота
REC_SEC = 10 #длина записи
OUTPUT = "files/output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FRT,channels=CHAN,rate=RT,input=True,frames_per_buffer=CHUNK) # открываем поток для записи
print("rec")
frames = [] # формируем выборку данных фреймов
for i in range(0, int(RT / CHUNK * REC_SEC)):
    data = stream.read(CHUNK)
    frames.append(data)
print("done")
stream.stop_stream() # останавливаем и закрываем поток
stream.close()
p.terminate()

w = wave.open(OUTPUT, 'wb')
w.setnchannels(CHAN)
w.setsampwidth(p.get_sample_size(FRT))
w.setframerate(RT)
w.writeframes(b''.join(frames))
w.close()

model_path = 'model/vosk-model-small-ru-0.22'

model = Model(model_path)

def transcribe_audio(file_path):
    # Откройте аудиофайл с помощью wave
    wf = wave.open(file_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)  # Чтобы получать слова вместо простых результатов

    # Чтение данных и распознавание
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            print("Процесс распознавания...")  # Логирование процесса

    # Получение итогового результата
    result = rec.FinalResult()
    return result

audio_file_path = 'files/output.wav'  # Замените на путь к вашему аудиофайлу
text = transcribe_audio(audio_file_path)
# print(text)

with open("files/transcribed_text.txt", "w") as text_file:
    text_file.write(text)
    print("Текст успешно сохранен в файл 'transcribed_text.txt'.")

with open("files/transcribed_text.txt", "r") as file:
    print(file.read())
