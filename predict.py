from finbert.finbert import predict
from finbert.utils import get_device
from transformers import AutoModelForSequenceClassification
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Sentiment analyzer")

parser.add_argument("-a", action="store_true", default=False)

parser.add_argument("--text_path", type=str, help="Path to the text file.")
parser.add_argument("--output_dir", type=str, help="Where to write the results")
parser.add_argument("--model_path", type=str, help="Path to classifier model")
parser.add_argument("--use_gpu", action="store_true", default=False,
                    help="Use GPU/MPS for inference (auto-detected if not specified)")
parser.add_argument("--no_gpu", action="store_true", default=False,
                    help="Force CPU usage even if GPU/MPS is available")

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


with open(args.text_path, "r") as f:
    text = f.read()

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_path, num_labels=3, cache_dir=None
)

# Automatically detect and use GPU/MPS if available (unless explicitly disabled)
if args.no_gpu:
    use_gpu = False
elif args.use_gpu:
    use_gpu = True
else:
    # Auto-detect: use GPU/MPS if available
    device = get_device(no_cuda=False)
    use_gpu = device.type != "cpu"
    if use_gpu:
        print(f"Auto-detected device: {device.type}, using GPU/MPS for inference")

output = "predictions.csv"
results = predict(
    text, model, write_to_csv=True, path=os.path.join(args.output_dir, output), use_gpu=use_gpu
)

# Print results in a human-readable format
print("\n" + "=" * 80)
print("SENTIMENT ANALYSIS RESULTS")
print("=" * 80 + "\n")

for idx, row in results.iterrows():
    # Extract probabilities from logit list
    probs = np.array(row["logit"])
    positive_prob = probs[0] * 100
    negative_prob = probs[1] * 100
    neutral_prob = probs[2] * 100

    # Get the predicted sentiment
    sentiment = row["prediction"].upper()

    # Truncate long sentences for display
    sentence = row["sentence"]
    if len(sentence) > 100:
        sentence = sentence[:97] + "..."

    print(f"[{idx + 1}] {sentiment}")
    print(f"    Sentence: {sentence}")
    print(
        f"    Confidence: Positive: {positive_prob:.1f}% | Negative: {negative_prob:.1f}% | Neutral: {neutral_prob:.1f}%"
    )
    print(f"    Sentiment Score: {row['sentiment_score']:.3f} (range: -1.0 to +1.0)")
    print()

# Print summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
sentiment_counts = results["prediction"].value_counts()
total_sentences = len(results)

print(f"\nTotal sentences analyzed: {total_sentences}")
print(f"\nSentiment distribution:")
for sentiment in ["positive", "negative", "neutral"]:
    count = sentiment_counts.get(sentiment, 0)
    percentage = (count / total_sentences) * 100
    print(f"  {sentiment.upper():8s}: {count:2d} ({percentage:5.1f}%)")

avg_sentiment_score = results["sentiment_score"].mean()
print(f"\nAverage sentiment score: {avg_sentiment_score:.3f}")
if avg_sentiment_score > 0.3:
    overall = "POSITIVE"
elif avg_sentiment_score < -0.3:
    overall = "NEGATIVE"
else:
    overall = "NEUTRAL"
print(f"Overall sentiment: {overall}")

print(f"\nResults saved to: {os.path.join(args.output_dir, output)}")
print("=" * 80 + "\n")
