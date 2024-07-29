# 시뮬레이션 결과 

df = read.csv("/Users/jimin/Library/CloudStorage/GoogleDrive-tonnonssi@gmail.com/My Drive/Minesweeper [RL]/firstReveal.csv")

df = df[-c(1)] # 필요없는 열 삭제 

head(df)

dim(df)
sum(df[1] != 1) / nrow(df) # vertex : 1개 이상 까질 확률 
sum(df[2] != 1) / nrow(df) # random : 1개 이상 까질 확률 


# 확률론적
choose(70,3) / choose(80,3) # vertex
choose(70,5) / choose(80,5) # 테두리
choose(70,8) / choose(80,8) # 중앙 
