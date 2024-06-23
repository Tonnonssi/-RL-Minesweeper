from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/update_board', methods=['POST'])

def update_board():
    data = request.json
    board_array = np.array(data['board'])
    print("Received board state:")
    print(board_array)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)