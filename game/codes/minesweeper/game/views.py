from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import numpy as np
from .serializers import BoardSerializer

@api_view(['POST'])
def update_board(request):
    serializer = BoardSerializer(data=request.data)
    if serializer.is_valid():
        board = serializer.validated_data['board']
        board_array = np.array(board)
        print("Received board state:")
        print(board_array)
        return Response({"status": "success"}, status=status.HTTP_200_OK)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
