package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.store.CosineSignature;
import com.davidbracewell.apollo.linalg.store.InMemoryLSH;
import com.davidbracewell.apollo.linalg.store.VectorStore;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.conversion.Convert;
import com.davidbracewell.stream.SparkStream;
import lombok.Getter;
import lombok.Setter;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import scala.collection.JavaConversions;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * <p>Wrapper around Spark's Word2Vec implementation</p>
 *
 * @author David B. Bracewell
 */
public class SparkWord2Vec extends EmbeddingLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int minCount = 5;
   @Getter
   @Setter
   private int numIterations = 1;
   @Getter
   @Setter
   private double learningRate = 0.025;
   @Getter
   @Setter
   private long randomSeed = new Date().getTime();
   @Getter
   @Setter
   private int windowSize = 5;

   @Override
   public void resetLearnerParameters() {

   }

   @Override
   protected Embedding trainImpl(Dataset<Sequence> dataset) {
      Word2Vec w2v = new Word2Vec();
      w2v.setMinCount(minCount);
      w2v.setVectorSize(getDimension());
      w2v.setLearningRate(learningRate);
      w2v.setNumIterations(numIterations);
      w2v.setWindowSize(getWindowSize());
      w2v.setSeed(randomSeed);
      SparkStream<Iterable<String>> sentences = new SparkStream<>(dataset.stream()
                                                                         .map(sequence -> {
                                                                            List<String> sentence = new ArrayList<>();
                                                                            for (Instance instance : sequence) {
                                                                               sentence.add(instance.getFeatures()
                                                                                                    .get(0)
                                                                                                    .getName());
                                                                            }
                                                                            return sentence;
                                                                         }));
      Word2VecModel model = w2v.fit(sentences.getRDD());

      Encoder encoder = new IndexEncoder();
      VectorStore<String> vectorStore = InMemoryLSH.builder()
                                                   .dimension(getDimension())
                                                   .signatureSupplier(CosineSignature::new)
                                                   .createVectorStore();
      for (Map.Entry<String, float[]> vector : JavaConversions.mapAsJavaMap(model.getVectors()).entrySet()) {
         encoder.encode(vector.getKey());
         vectorStore.add(vector.getKey(), new DenseVector(Convert.convert(vector.getValue(), double[].class)));
      }
      return new Embedding(new EncoderPair(dataset.getLabelEncoder(), encoder), vectorStore);
   }


}// END OF SparkWord2Vec
