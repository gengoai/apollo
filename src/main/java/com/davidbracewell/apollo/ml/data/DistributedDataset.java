package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.apollo.ml.encoder.Encoder;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.encoder.LabelEncoder;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.SparkStream;
import lombok.NonNull;

import java.util.Random;

/**
 * <p>A distributed dataset using Apache Spark.</p>
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
public class DistributedDataset<T extends Example> extends Dataset<T> {
   private static final long serialVersionUID = 1L;
   private volatile MStream<T> stream;

   /**
    * Instantiates a new Distributed Dataset.
    *
    * @param featureEncoder the feature encoder
    * @param labelEncoder   the label encoder
    * @param preprocessors  the preprocessors
    */
   protected DistributedDataset(Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      super(featureEncoder, labelEncoder, preprocessors);
   }

   @Override
   protected void addAll(@NonNull MStream<T> otherStream) {
      if (this.stream == null) {
         this.stream = new SparkStream<>(otherStream.cache());
      } else {
         this.stream = this.stream.union(otherStream);
      }
   }

   @Override
   public Dataset<T> cache() {
      this.stream = stream.cache();
      return this;
   }

   @Override
   public void close() throws Exception {
      this.stream.close();
   }

   @Override
   protected Dataset<T> create(MStream<T> instances, Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      DistributedDataset<T> dataset = new DistributedDataset<>(featureEncoder, labelEncoder, preprocessors);
      dataset.stream = new SparkStream<>(instances);
      return dataset;
   }

   @Override
   public DatasetType getType() {
      return DatasetType.Distributed;
   }

   @Override
   public Dataset<T> mapSelf(@NonNull SerializableFunction<? super T, T> function) {
      this.stream = stream.map(function);
      return this;
   }

   @Override
   public Dataset<T> shuffle(Random random) {
      return create(stream.shuffle(), getFeatureEncoder(), getLabelEncoder(), getPreprocessors());
   }

   @Override
   public int size() {
      return (int) stream.count();
   }

   @Override
   public MStream<T> stream() {
      return stream;
   }
}// END OF DistributedDataset
