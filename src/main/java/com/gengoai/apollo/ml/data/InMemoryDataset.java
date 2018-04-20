package com.gengoai.apollo.ml.data;

import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.encoder.Encoder;
import com.gengoai.apollo.ml.encoder.LabelEncoder;
import com.gengoai.apollo.ml.preprocess.PreprocessorList;
import com.gengoai.Interner;
import com.gengoai.collection.Collect;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableFunction;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import lombok.NonNull;

import java.util.*;

/**
 * <p>A Dataset that stores all examples in memory. Feature names are interned to conserve memory. In addition, methods
 * for adding examples are public.</p>
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
public class InMemoryDataset<T extends Example> extends Dataset<T> {
   private static final Interner<String> interner = new Interner<>();
   private static final long serialVersionUID = 1L;
   private final List<T> instances = new ArrayList<>();

   /**
    * Instantiates a new In-memory dataset.
    *
    * @param featureEncoder the feature encoder
    * @param labelEncoder   the label encoder
    * @param preprocessors  the preprocessors
    */
   public InMemoryDataset(Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      super(featureEncoder, labelEncoder, preprocessors);
   }

   /**
    * Adds an example to the dataset
    *
    * @param example the example
    */
   public void add(T example) {
      if (example != null) {
         getLabelEncoder().encode(example.getLabelSpace());
         instances.add(example);
      }
   }

   @Override
   protected void addAll(@NonNull Collection<T> instances) {
      super.addAll(instances);
   }

   @Override
   protected void addAll(@NonNull MStream<T> stream) {
      for (T instance : Collect.asIterable(stream.iterator())) {
         getLabelEncoder().encode(instance.getLabelSpace());
         instances.add(Cast.as(instance.intern(interner)));
      }
   }

   @Override
   public void close() {
      instances.clear();
   }

   @Override
   protected Dataset<T> create(@NonNull MStream<T> instances, @NonNull Encoder featureEncoder, @NonNull LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      InMemoryDataset<T> dataset = new InMemoryDataset<>(featureEncoder, labelEncoder, preprocessors);
      dataset.addAll(instances);
      return dataset;
   }

   @Override
   public DatasetType getType() {
      return DatasetType.InMemory;
   }


   @Override
   public Iterator<T> iterator() {
      return instances.iterator();
   }

   @Override
   public Dataset<T> mapSelf(@NonNull SerializableFunction<? super T, T> function) {
      for (int i = 0; i < instances.size(); i++) {
         instances.set(i, function.apply(instances.get(i)));
      }
      return this;
   }

   @Override
   public Dataset<T> shuffle(@NonNull Random random) {
      Collections.shuffle(instances, random);
      return this;
   }

   @Override
   public int size() {
      return instances.size();
   }

   @Override
   public Dataset<T> slice(int start, int end) {
      return create(StreamingContext.local().stream(instances.subList(start, end).stream()));
   }

   @Override
   public Spliterator<T> spliterator() {
      return instances.spliterator();
   }

   @Override
   public MStream<T> stream() {
      return StreamingContext.local().stream(instances);
   }
}// END OF InMemoryDataset
