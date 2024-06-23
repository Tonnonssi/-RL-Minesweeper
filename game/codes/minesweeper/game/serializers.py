# game/serializers.py
from rest_framework import serializers

class BoardSerializer(serializers.Serializer):
    board = serializers.ListField(child=serializers.ListField(child=serializers.DictField()))
