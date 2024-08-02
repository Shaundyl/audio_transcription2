from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from .models import Transcription, Speaker
from .serializers import TranscriptionSerializer
import torch
from pyannote.audio import Pipeline
import whisper
from pydub import AudioSegment
import numpy as np
import re

class TranscriptionViewSet(viewsets.ModelViewSet):
    queryset = Transcription.objects.all()
    serializer_class = TranscriptionSerializer

    @action(detail=True, methods=['post'])
    def process(self, request, pk=None):
        transcription = self.get_object()
        
        if transcription.processed:
            return Response({"message": "This audio has already been processed."}, status=status.HTTP_400_BAD_REQUEST)

        # Your existing code for processing the audio file
        hugging_face_token = "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"
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hugging_face_token)
        whisper_model = whisper.load_model("base")

        audio_file = transcription.audio_file.path
        audio = AudioSegment.from_wav(audio_file)
        diarization = pipeline(audio_file)

        def transcribe_segment(audio_file, start, end):
            audio = whisper.load_audio(audio_file)
            audio_segment = audio[int(start * 16000):int(end * 16000)]
            result = whisper_model.transcribe(audio_segment, language="english")
            return result["text"]

        def extract_name(introduction):
            patterns = [r"my name is (\w+)", r"i am (\w+)", r"this is (\w+)"]
            for pattern in patterns:
                match = re.search(pattern, introduction, re.IGNORECASE)
                if match:
                    return match.group(1)
            return None

        results = []
        speaker_names = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            transcription_text = transcribe_segment(audio_file, turn.start, turn.end)
            
            if speaker not in speaker_names:
                name = extract_name(transcription_text)
                if name:
                    speaker_names[speaker] = name
                else:
                    speaker_names[speaker] = speaker

            results.append((speaker_names[speaker], transcription_text))

        # Save results to the database
        for speaker, text in results:
            Speaker.objects.create(transcription=transcription, name=speaker, text=text)

        transcription.processed = True
        transcription.save()

        return Response({"message": "Audio processed successfully."}, status=status.HTTP_200_OK)