package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.encoder.Encoder;
import com.gengoai.apollo.ml.encoder.LabelEncoder;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.function.SerializableFunction;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import lombok.NonNull;

import java.util.Random;

/**
 * <p>A dataset wrapping a Mango Stream.</p>
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public class MStreamDataset<T extends Example> extends Dataset<T> {
   private static final long serialVersionUID = 1L;
   private volatile MStream<T> stream;

   /**
    * Instantiates a new MStream Dataset.
    *
    * @param featureEncoder the feature encoder
    * @param labelEncoder   the label encoder
    * @param preprocessors  the preprocessors
    * @param stream         the stream
    */
   public MStreamDataset(Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors, MStream<T> stream) {
      super(featureEncoder, labelEncoder, preprocessors);
      this.stream = stream;
   }

   @Override
   protected void addAll(MStream<T> stream) {
      if (this.stream == null) {
         this.stream = stream;
      } else {
         this.stream = this.stream.union(stream);
      }
   }

   @Override
   protected Dataset<T> create(MStream<T> instances, Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      return new MStreamDataset<>(featureEncoder, labelEncoder, preprocessors, instances);
   }

   @Override
   public StreamingContext getStreamingContext() {
      return stream.getContext();
   }

   @Override
   public DatasetType getType() {
      return DatasetType.Stream;
   }

   @Override
   public Dataset<T> mapSelf(@NonNull SerializableFunction<? super T, T> function) {
      stream = stream.map(function);
      return this;
   }

   @Override
   public Dataset<T> shuffle(Random random) {
      return create(stream.shuffle(random),
                    getFeatureEncoder(),
                    getLabelEncoder(),
                    getPreprocessors());
   }

   @Override
   public int size() {
      return (int) stream.count();
   }

   @Override
   public MStream<T> stream() {
      return stream;
   }
}// END OF MStreamDataset
