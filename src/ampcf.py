import torch
import torch.nn as nn
import torch.nn.functional as F


class AMPCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_personas: int,
        embedding_dim,
        attention_dim,
    ):
        super(AMPCF, self).__init__()
        self.num_personas = num_personas
        self.user_embedding = nn.Embedding(num_users, embedding_dim * num_personas)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        self.Au = nn.Linear(embedding_dim, attention_dim)
        self.Av = nn.Linear(embedding_dim, attention_dim)

    def forward(self, user_ids: int, item_ids: int) -> (torch.Tensor, torch.Tensor):
        user_embedding = self.user_embedding(user_ids)  # Shape: (batch_size, embedding_dim * num_personas)
        item_embedding = self.item_embedding(item_ids)  # Shape: (batch_size, embedding_dim)
        item_bias = self.item_bias(item_ids).squeeze()  # Shape: (batch_size,)

        # Reshape user_embedding to separate personas
        user_embedding = user_embedding.view(
            -1, self.num_personas, user_embedding.size(1) // self.num_personas
        )  # Shape: (batch_size, num_personas, embedding_dim)

        # Compute attention scores
        attention_scores = torch.empty(user_embedding.size(0), self.num_personas)
        for k in range(self.num_personas):
            psi_ik = self.Au(user_embedding[:, k, :])  # Shape: (batch_size, attention_dim)
            phi_j = self.Av(item_embedding)  # Shape: (batch_size, attention_dim)
            attention_scores[:, k] = F.cosine_similarity(psi_ik, phi_j, dim=1)

        # Normalize attention scores
        attention_scores = F.softmax(attention_scores, dim=1).to(user_embedding.device)

        # Compute attentive user-item vector
        x_ij = torch.sum(attention_scores.unsqueeze(-1) * user_embedding, dim=1)  # Shape: (batch_size, embedding_dim)

        # Compute similarity score
        s_ij = (x_ij * item_embedding).sum(dim=-1)  # Shape: (batch_size,)
        y_ij = s_ij + item_bias

        return y_ij, attention_scores

    def negative_sampling(self, user_ids: torch.Tensor, item_ids: torch.Tensor, N: int) -> torch.Tensor:
        # TODO: Sample negative items from unigram item distribution
        # N is the number of negative samples per positive pair
        # Return a tensor of shape (num_positive_pairs * N, 2)

        all_items = torch.arange(self.item_embedding.num_embeddings, device=item_ids.device)
        negative_items = all_items[~torch.isin(all_items, item_ids)]
        negative_samples = negative_items[torch.randint(0, len(negative_items), (item_ids.size(0) * N,))]

        return torch.stack([user_ids.repeat_interleave(N), negative_samples], dim=1)

    def cal_loss(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        N: int,
        lambda_p: float,
        lambda_n: float,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        negative_pairs = self.negative_sampling(user_ids, item_ids, N)  # Shape: (num_positive_pairs * N, 2)
        positive_scores, positive_attention_scores = self.forward(user_ids, item_ids)
        negative_scores, negative_attention_scores = self.forward(negative_pairs[:, 0], negative_pairs[:, 1])

        # Compute the negative log-likelihood loss
        L_ij_D = -positive_scores + torch.log(torch.exp(positive_scores).sum() + torch.exp(negative_scores).sum())
        L_ij_D = L_ij_D.mean()

        # Compute the entropy loss for positive and negative examples
        H_ij_p = -torch.sum(
            positive_attention_scores * torch.log(positive_attention_scores + 1e-8),
            dim=1,
        ).mean()
        H_ij_n = -torch.sum(
            negative_attention_scores * torch.log(negative_attention_scores + 1e-8),
            dim=1,
        ).mean()

        L = alpha * L_ij_D + (1 - alpha) * (lambda_p * H_ij_p - lambda_n * H_ij_n)

        return L
