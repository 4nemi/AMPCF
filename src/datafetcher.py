from torch.utils.data import Dataset
import pandas as pd
import os

from models import DataContainer


class TrainDataset(Dataset):
    def __init__(self, positive_pairs: pd.DataFrame):
        self.user_ids = positive_pairs["user_id"].values
        self.item_ids = positive_pairs["movie_id"].values

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx]


class DataFetcher:
    def __init__(self, num_test_items: int, dataset_name: str):
        self.num_test_items = num_test_items
        self.dataset_name = dataset_name
        self.data_path = os.path.join("../data", dataset_name)

    def load(self) -> Dataset:
        # TODO: validに関しても同様にranking用の評価データを作成する
        # データの読み込み
        ratings, movies = self._load_movie_lens_1m()
        # データの分割
        train, valid, test = self._split_data(ratings)
        # ranking用の評価データは、ユーザーが高評価したアイテムのみを抽出（4以上）かつ、ratingの高い順にソート
        valid_user2items = (
            valid[valid["rating"] >= 4]
            .sort_values(["user_id", "rating"], ascending=[True, False])
            .groupby("user_id")["movie_id"]
            .agg(list)
            .to_dict()
        )

        test_user2items = (
            test[test["rating"] >= 4]
            .sort_values(["user_id", "rating"], ascending=[True, False])
            .groupby("user_id")["movie_id"]
            .agg(list)
            .to_dict()
        )
        train = TrainDataset(train)
        return DataContainer(train, valid, test, valid_user2items, test_user2items, movies)

    def _split_data(self, ratings: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        # Train, Valid, Testに分割
        # ユーザーごとに最新5件をTest, 5件前から10件前までをValid, それ以前をTrainとする
        ratings["rating_order"] = ratings.groupby("user_id")["timestamp"].rank(method="first", ascending=False)
        train = ratings[ratings["rating_order"] > self.num_test_items * 2]
        valid = ratings[
            (ratings["rating_order"] <= self.num_test_items * 2) & (ratings["rating_order"] > self.num_test_items)
        ]
        test = ratings[ratings["rating_order"] <= self.num_test_items]
        return train, valid, test

    def _load_movie_lens_1m(self) -> (pd.DataFrame, pd.DataFrame):
        # TODO: Movielens genomeを使った実装
        # movieデータの読み込み
        m_cols = ["movie_id", "title", "genre"]
        movies = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"),
            sep="::",
            names=m_cols,
            encoding="latin-1",
            engine="python",
        )

        # genreをlist形式で保持
        movies["genre"] = movies["genre"].apply(lambda x: x.split("|"))

        # 評価データの読み込み
        r_cols = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_csv(
            os.path.join(self.data_path, "ratings.dat"),
            sep="::",
            names=r_cols,
            encoding="latin-1",
            engine="python",
        )

        # 2つ未満の評価を持つユーザーを削除
        ratings = ratings.groupby("user_id").filter(lambda x: len(x) >= 2)

        # user_idをふり直す
        ratings["user_id"] = ratings.groupby("user_id").ngroup()
        # movie_idをふり直す
        ratings["old_movie_id"] = ratings["movie_id"]
        ratings["movie_id"] = ratings.groupby("movie_id").ngroup()

        movie_id_map = dict(zip(ratings["old_movie_id"], ratings["movie_id"]))
        movies["movie_id"] = movies["movie_id"].map(movie_id_map)

        # 必要なカラムのみを抽出
        # positive_pairs = ratings[["user_id", "movie_id", "timestamp"]]

        return ratings, movies
