from rest_framework import views, status
from rest_framework.response import Response
from .models import Transcription, Speaker
from .serializers import TranscriptionSerializer
import torch
from pyannote.audio import Pipeline
import whisper
from pydub import AudioSegment
import numpy as np
import re

class TranscriptionView(views.APIView):
    def post(self, request):
        audio_file = request.FILES.get('audio_file')
        if not audio_file:
            return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save the audio file
        transcription = Transcription.objects.create(audio_file=audio_file)

        # Process the audio file
        hugging_face_token = "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"
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hugging_face_token)
        whisper_model = whisper.load_model("base")

        audio_file_path = transcription.audio_file.path
        diarization = pipeline(audio_file_path)

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
        full_transcript = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            transcription_text = transcribe_segment(audio_file_path, turn.start, turn.end)
            
            if speaker not in speaker_names:
                name = extract_name(transcription_text)
                if name:
                    speaker_names[speaker] = name
                else:
                    speaker_names[speaker] = f"Speaker {len(speaker_names) + 1}"

            results.append((speaker_names[speaker], transcription_text))
            full_transcript.append(f"{speaker_names[speaker]}: {transcription_text}")

        # Save results to the database
        for speaker, text in results:
            Speaker.objects.create(transcription=transcription, name=speaker, text=text)

        transcription.transcript = "\n".join(full_transcript)
        transcription.save()

        serializer = TranscriptionSerializer(transcription)
        return Response(serializer.data, status=status.HTTP_200_OK)