#!/usr/bin/env python3

import json
from http import HTTPStatus
from typing import Tuple, TypedDict, Union, List
from enum import Enum
import re
import audioop
import tempfile
import logging

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer, SetLogLevel
import wave


FRAMERATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# You can set log level to -1 to disable debug messages
SetLogLevel(-1)

MODEL_NAME = "models/vosk-model-small-ru-0.22"

model = Model(MODEL_NAME)
logger.info(f"model {MODEL_NAME} loaded")

recognizer = KaldiRecognizer(model, FRAMERATE)
recognizer.SetWords(True)
logging.info(f"speech recognizer initialized with framerate={FRAMERATE}")


class ActionType(str, Enum):
    LIGHT_ON = "light_on"
    LIGHT_OFF = "light_off"
    CHANGE_LIGHTNESS = "change_lightness"
    UNRECOGNIZED = "unrecognized"


class SwitchLightCommand(TypedDict):
    action: ActionType
    light_num: int


class SwitchLightnessCommand(TypedDict):
    action: ActionType
    light_num: int
    lightness_level: int


class UnrecognizedCommand(TypedDict):
    action: ActionType


Command = Union[SwitchLightCommand, SwitchLightnessCommand, UnrecognizedCommand]


def read_file_wav(file: UploadFile) -> Tuple[bytes, int]:
    with wave.open(file.file, "rb") as wf:
        if wf.getnchannels() != CHANNELS or wf.getsampwidth() != SAMPLE_WIDTH or wf.getcomptype() != "NONE":
            # raise HTTPException(
            #     status_code=HTTPStatus.BAD_REQUEST,
            #     detail="Audio file must be WAV format mono PCM.",
            # )
            raise ValueError("Wrong wav file")

        framerate = wf.getframerate()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)

    return data, framerate


def read_file_audiodata(file: UploadFile) -> Tuple[bytes, int]:
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav")

    autiodata = file.file.read()

    with wave.open(temp_file, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(FRAMERATE)
        wf.writeframes(autiodata)

    with wave.open(temp_file, "rb") as wf:
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        data = wf.readframes(nframes)

    temp_file.close()

    return data, framerate


def convert(fragment: bytes, orig_rate: int) -> bytes:
    converted = audioop.ratecv(fragment, SAMPLE_WIDTH, CHANNELS, orig_rate, FRAMERATE, None)
    return converted[0]


def speech_recognize(_input: bytes) -> str:
    recognizer.AcceptWaveform(_input)
    res = recognizer.FinalResult()

    d = json.loads(res)
    text = d["text"]

    return text


def convert_ligth_num_to_num(s: str) -> int:
    c = {
        "один": 1,
        "два": 2,
        "три": 3,
        "четыре": 4,
    }
    try:
        return c[s]
    except KeyError:
        raise ValueError("Wrong light num level")


def convert_level_to_num(parts: List[str]) -> int:
    decs = {
        "десять": 10,
        "двадцать": 20,
        "тридцать": 30,
        "сорок": 40,
        "пятьдесят": 50,
        "шестьдесят": 60,
        "семьдесят": 70,
        "восемьдесят": 80,
        "девяносто": 90,
        "сто": 100,
    }
    units = {
        "один": 1,
        "два": 2,
        "три": 3,
        "четыре": 4,
        "пять": 5,
        "шесть": 6,
        "семь": 7,
        "восемь": 8,
        "девять": 9,
    }
    try:
        if len(parts) == 1:
            if parts[0] in decs:
                return decs[parts[0]]
            elif parts[0] in units:
                return units[parts[0]]
        elif len(parts) == 2:
            return decs[parts[0]] + units[parts[1]]
        else:
            raise ValueError("Wrong lightness level")
    except KeyError:
        raise ValueError("Wrong lightness level")


def analize_command(text: str) -> Command:
    logging.info("recognized text:", text)
    try:
        if re.match("^включить лампочку \w{1,}$", text):
            light_num = convert_ligth_num_to_num(text.split(" ")[-1])
            return SwitchLightCommand(
                action=ActionType.LIGHT_ON,
                light_num=light_num
            )
        elif re.match("^выключить лампочку \w{1,}$", text):
            light_num = convert_ligth_num_to_num(text.split(" ")[-1])
            return SwitchLightCommand(
                action=ActionType.LIGHT_OFF,
                light_num=light_num
            )
        elif re.match("^установить яркость лампочки \w{1,} в \w{1,} \w{1,}", text):
            light_num = convert_ligth_num_to_num(text.split(" ")[3])
            lightness_level = convert_level_to_num(text.split(" ")[5:7])
            return SwitchLightnessCommand(
                action=ActionType.CHANGE_LIGHTNESS,
                light_num=light_num,
                lightness_level=lightness_level,
            )
        else:
            return UnrecognizedCommand(action=ActionType.UNRECOGNIZED)
    except ValueError:
        return UnrecognizedCommand(action=ActionType.UNRECOGNIZED)


@app.post("/recognize")
def recognize(file: UploadFile):
    try:
        data, framerate = read_file_wav(file)
    except Exception:
        data, framerate = read_file_audiodata(file)

    converted = convert(data, framerate)
    text = speech_recognize(converted)
    command = analize_command(text)

    return {"filename": file.filename, "text": text, "command": command}
