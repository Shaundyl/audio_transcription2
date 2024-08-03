from rest_framework import serializers
from .models import Transcription, Speaker

class SpeakerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Speaker
        fields = ['name', 'text']

class TranscriptionSerializer(serializers.ModelSerializer):
    speakers = SpeakerSerializer(many=True, read_only=True)

    class Meta:
        model = Transcription
        fields = ['id', 'audio_file', 'transcript', 'created_at', 'speakers']