package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linalg.CosineSignature;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.InMemoryLSH;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.VectorStore;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.DatasetType;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.preprocess.transform.TFIDFTransform;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.config.Config;
import com.davidbracewell.conversion.Convert;
import com.davidbracewell.stream.SparkStream;
import com.davidbracewell.stream.StreamingContext;
import com.davidbracewell.string.StringUtils;
import lombok.Getter;
import lombok.Setter;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import scala.collection.JavaConversions;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class SparkWord2Vec extends EmbeddingLearner {
  private static final long serialVersionUID = 1L;

  private
  @Getter
  @Setter
  int minCount = 5;
  private
  @Getter
  @Setter
  int windowSize = 5;
  private
  @Getter
  @Setter
  int dimension = 100;
  private
  @Getter
  @Setter
  int numIterations = 1;
  private
  @Getter
  @Setter
  double learningRate = 0.025;

  public static void main(String[] args) {
    Config.initialize("");
    Config.setProperty("spark.master", "local[*]");
    Dataset<Sequence> sentences = Dataset.embedding(
      DatasetType.Distributed,
      StreamingContext.distributed().textFile("/home/david/prj/text-analysis/sentences.txt"),
      line -> Stream.of(line.split("\\s+")).map(String::toLowerCase).filter(w -> StringUtils.hasLetter(w) && w.length() > 2)
    ).preprocess(PreprocessorList.create(new TFIDFTransform().asSequenceProcessor()));
    SparkLSA word2Vec = new SparkLSA();
    Embedding embedding = word2Vec.train(sentences);
    System.out.println(embedding.similarity("flavors", "spicy"));
    embedding.nearest($("flavors", "smell"), $("spicy"), 10, 0).forEach(slv -> System.out.println(slv.<String>getLabel() + " : " + slv.getScore()));
  }

  @Override
  protected Embedding trainImpl(Dataset<Sequence> dataset) {
    Word2Vec w2v = new Word2Vec();
    w2v.setMinCount(minCount);
    w2v.setVectorSize(dimension);
    w2v.setLearningRate(learningRate);
    w2v.setNumIterations(numIterations);
    w2v.setWindowSize(windowSize);
    SparkStream<Iterable<String>> sentences = new SparkStream<>(
      dataset.stream().map(sequence -> {
        List<String> sentence = new ArrayList<>();
        for (Instance instance : sequence) {
          sentence.add(instance.getFeatures().get(0).getName());
        }
        return sentence;
      })
    );
    Word2VecModel model = w2v.fit(sentences.asRDD());

    Encoder encoder = new IndexEncoder();
    VectorStore<String> vectorStore = InMemoryLSH.builder()
      .dimension(dimension)
      .signatureSupplier(CosineSignature::new)
      .createVectorStore();
    for (Map.Entry<String, float[]> vector : JavaConversions.mapAsJavaMap(model.getVectors()).entrySet()) {
      encoder.encode(vector.getKey());
      vectorStore.add(new LabeledVector(vector.getKey(), new DenseVector(Convert.convert(vector.getValue(), double[].class))));
    }
    return new Embedding(new EncoderPair(dataset.getLabelEncoder(), encoder), vectorStore);
  }

  @Override
  public void reset() {

  }


}// END OF SparkWord2Vec
