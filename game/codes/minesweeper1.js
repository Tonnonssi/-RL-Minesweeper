const numRows = 9;
const numCols = 9;
const numMines = 10;

// MineSweeperGame 클래스 정의
class MineSweeperGame {
    constructor(numRows, numCols, numMines) {
        this.numRows = numRows;
        this.numCols = numCols;
        this.numMines = numMines;
        this.board = [];
        this.gameBoard = document.getElementById("gameBoard");
        this.initializeBoard();
        this.renderBoard();
    }

    initializeBoard() {
        // Initialize board
        for (let i = 0; i < this.numRows; i++) {
            this.board[i] = [];
            for (let j = 0; j < this.numCols; j++) {
                this.board[i][j] = {
                    isMine: false,
                    revealed: false,
                    count: 0,
                };
            }
        }

        // Place mines randomly
        let minesPlaced = 0;
        while (minesPlaced < this.numMines) {
            const row = Math.floor(Math.random() * this.numRows);
            const col = Math.floor(Math.random() * this.numCols);
            if (!this.board[row][col].isMine) {
                this.board[row][col].isMine = true;
                minesPlaced++;
            }
        }

        // Calculate counts
        for (let i = 0; i < this.numRows; i++) {
            for (let j = 0; j < this.numCols; j++) {
                if (!this.board[i][j].isMine) {
                    let count = 0;
                    for (let dx = -1; dx <= 1; dx++) {
                        for (let dy = -1; dy <= 1; dy++) {
                            const ni = i + dx;
                            const nj = j + dy;
                            if (ni >= 0 && ni < this.numRows &&
                                nj >= 0 && nj < this.numCols &&
                                this.board[ni][nj].isMine) {
                                count++;
                            }
                        }
                    }
                    this.board[i][j].count = count;
                }
            }
        }
    }

    revealCell(row, col) {
        if (row < 0 || row >= this.numRows || col < 0 || col >= this.numCols || this.board[row][col].revealed) {
            return;
        }

        this.board[row][col].revealed = true;

        if (this.board[row][col].isMine) {
            // Handle game over
            alert("Game Over! You stepped on a mine.");
        } else if (this.board[row][col].count === 0) {
            // If cell has no mines nearby,
            // Reveal adjacent cells
            for (let dx = -1; dx <= 1; dx++) {
                for (let dy = -1; dy <= 1; dy++) {
                    this.revealCell(row + dx, col + dy);
                }
            }
        }

        this.renderBoard();
    }

    renderBoard() {
        this.gameBoard.innerHTML = "";

        for (let i = 0; i < this.numRows; i++) {
            for (let j = 0; j < this.numCols; j++) {
                const cell = document.createElement("div");
                cell.className = "cell";

                if (this.board[i][j].revealed) {
                    cell.classList.add("revealed");

                    if (this.board[i][j].isMine) {
                        cell.classList.add("mine");
                        cell.textContent = "M";
                    } else if (this.board[i][j].count > 0) {
                        cell.textContent = this.board[i][j].count;
                    }
                }

                cell.addEventListener("click", () => this.revealCell(i, j));
                this.gameBoard.appendChild(cell);
            }
            this.gameBoard.appendChild(document.createElement("br"));
        }
    }
}

// 게임 초기화 및 시작
const game = new MineSweeperGame(numRows, numCols, numMines);

