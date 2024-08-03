from django.db import models

class Transcription(models.Model):
    audio_file = models.FileField(upload_to='audio_files/')
    transcript = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Speaker(models.Model):
    transcription = models.ForeignKey(Transcription, related_name='speakers', on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    text = models.TextField()

    def __str__(self):
        return f"{self.name}: {self.text[:50]}..."