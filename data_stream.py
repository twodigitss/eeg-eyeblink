# Used for streaming EEG data via bluetooth.
from pylsl import StreamInlet, resolve_streams
from time import sleep

# For creating a CSV file.
import pandas as pd


def eeg_data_to_csv():
    # Finds all active LSL (Lab Streaming Layer) streams.
    streams = resolve_streams()

    # Loops through each stream 's' in 'streams'.
    # if s.type() is EEG, 'next' gets the first match.
    eeg_streams = next(s for s in streams if s.type() == 'EEG')

    # Connects the code to the EEG data stream.
    inlet = StreamInlet(eeg_streams)

    # Rows consisting of [timestamp, TP9, AF7, AF8, TP10
    eeg_data = []

    # Main data collection loop.
    # The underscore indicates repetition in for loop.
    for _ in range(20352):
        # Pulls sample and timestamp from the EEG data stream.
        sample, timestamp = inlet.pull_sample()

        # Saves each row.
        eeg_data.append([timestamp, sample[0], sample[1],
                         sample[2], sample[3]])

        # Optional delay for data collection.
        sleep(0)

    # Creates a pandas DataFrame
    df = pd.DataFrame(eeg_data, columns=['timestamp', 'TP9', 'AF7',
                                         'AF8', 'TP10'])

    # Saves the DataFrame to a CSV file format
    df.to_csv('raw_eeg_data.csv', index=False)

    print("EEG data saved to raw_eeg_data.csv")

def main():
    eeg_data_to_csv()

if __name__ == "__main__":
    main()
