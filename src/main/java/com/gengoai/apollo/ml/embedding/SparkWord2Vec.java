package com.gengoai.apollo.ml.embedding;

import com.gengoai.apollo.linear.store.VSBuilder;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.encoder.Encoder;
import com.gengoai.apollo.ml.encoder.IndexEncoder;
import com.gengoai.apollo.ml.sequence.Sequence;
import com.gengoai.stream.SparkStream;
import lombok.Getter;
import lombok.Setter;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

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
   @Getter
   @Setter
   private boolean fastKNN = false;

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
                                                                                                    .getFeatureName());
                                                                            }
                                                                            return sentence;
                                                                         }));
      Word2VecModel model = w2v.fit(sentences.getRDD());
      Encoder encoder = new IndexEncoder();
      VSBuilder builder;
//      if (fastKNN) {
//         builder = LSHVectorStore.<String>builder().signature("COSINE");
////      } else {
//         builder = InMemoryVectorStore.builder();
////      }
////      builder.dimension(getDimension());
////      builder.measure(Similarity.Cosine);
//      for (Map.Entry<String, float[]> vector : JavaConversions.mapAsJavaMap(model.getVectors()).entrySet()) {
//         encoder.encode(vector.getKey());
//         builder.add(vector.getKey(), NDArrayFactory.columnVector(vector.getValue()));
//      }
//      try {
//         return new Embedding(builder.build());
//      } catch (IOException e) {
//         throw new RuntimeException(e);
//      }
      return null;
   }


}// END OF SparkWord2Vec
