import apache_beam as beam
import re

inputs_pattern = 'data/*'
outputs_prefix = 'outputs/part'

# Running locally in the DirectRunner.
with beam.Pipeline() as pipeline:
  # Store the word counts in a PCollection.
  # Each element is a tuple of (word, count) of types (str, int).
  word_counts = (
      # The input PCollection is an empty pipeline.
      pipeline

      # Read lines from a text file.
      | 'Read lines' >> beam.io.ReadFromText(inputs_pattern)
      # Element type: str - text line

      # Use a regular expression to iterate over all words in the line.
      # FlatMap will yield an element for every element in an iterable.
      | 'Find words' >> beam.FlatMap(lambda line: re.findall(r"[a-zA-Z']+", line))
      # Element type: str - word

      # Create key-value pairs where the value is 1, this way we can group by
      # the same word while adding those 1s and get the counts for every word.
      | 'Pair words with 1' >> beam.Map(lambda word: (word, 1))
      # Element type: (str, int) - key: word, value: 1

      # Group by key while combining the value using the sum() function.
      | 'Group and sum' >> beam.CombinePerKey(sum)
      # Element type: (str, int) - key: word, value: counts
  )

  # We can process a PCollection through other pipelines too.
  (
      # The input PCollection is the word_counts created from the previous step.
      word_counts

      # Format the results into a string so we can write them to a file.
      | 'Format results' >> beam.Map(lambda word_count: str(word_count))
      # Element type: str - text line

      # Finally, write the results to a file.
      | 'Write results' >> beam.io.WriteToText(outputs_prefix)
  )

# Sample the first 20 results, remember there are no ordering guarantees.
run('head -n 20 {}-00000-of-*'.format(outputs_prefix))
