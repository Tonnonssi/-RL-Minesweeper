palette <- c("random"="#FF3366",
             "vertex"="#6699FF")

df = read.csv("/Users/jimin/Library/CloudStorage/GoogleDrive-tonnonssi@gmail.com/My Drive/Minesweeper [RL]/firstReveal.csv")

df = df[-c(1)] # 필요없는 열 삭제 

head(df)
df_long <- df %>% 
    pivot_longer(cols = c(vertex, random), names_to = "variable", values_to = "value")

# x를 범주형 데이터로 변환
df_long$X <- as.factor(df_long$X)

# 중앙값 계산
medians <- df_long %>%
    group_by(variable) %>%
    summarize(median = median(value), .groups = 'drop')

# 원본 데이터에 중앙값 추가
df_long <- df_long %>%
    left_join(medians, by = "variable")

# ggplot 사용하여 박스플롯 생성
ggplot(df_long, aes(x = variable, y = value)) +
    geom_violin(aes(fill = variable)) +  # 바이올린 플롯
    geom_boxplot(fill = "white", color = "black", width=0.05, alpha=0.5) +  # 박스플롯
    geom_hline(yintercept = 1, linetype = "dashed") +  # y = 1 수평선
    geom_hline(yintercept = 81, linetype = "dashed") +  # y = 81 수평선
    geom_text(data = medians, aes(x = variable, label = median, y = median), vjust = -0.5) +  # 중앙값 텍스트
    theme_minimal() +
    labs(x = "Type", y = "Count", title = "Number of Cells Revealed by Selecting the First Tile: Comparison of Vertex and Random Methods") + 
    scale_fill_manual(values=palette)
 
