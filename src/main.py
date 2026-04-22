import dloader
import experiments
import results


def main():
    print("\n=======================================")
    print("   MOVIE REVIEW SENTIMENT AI (RNN)")
    print("=======================================")

    print("\nLoading data...")
    data = dloader.read()
    print("Data loaded successfully.")

    print("\nRunning experiments...")
    experiments.run(data)

    print("\nGenerating evaluation outputs...")
    results.evaluate(data)

    print("\nDone.")


if __name__ == "__main__":
    main()