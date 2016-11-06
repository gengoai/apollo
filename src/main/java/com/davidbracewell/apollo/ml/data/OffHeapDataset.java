package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.LabelEncoder;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.Collect;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import com.davidbracewell.tuple.Tuples;
import com.google.common.base.Throwables;
import lombok.NonNull;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

import static com.davidbracewell.function.Unchecked.function;

/**
 * @author David B. Bracewell
 */
public class OffHeapDataset<T extends Example> extends Dataset<T> {
   private static final long serialVersionUID = 1L;
   private final AtomicLong id = new AtomicLong();
   private Resource outputResource = Resources.temporaryDirectory();
   private Class<T> clazz;
   private int size = 0;

   protected OffHeapDataset(Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      super(featureEncoder, labelEncoder, preprocessors);
      outputResource.deleteOnExit();
   }

   @Override
   protected void addAll(@NonNull MStream<T> instances) {
      long binSize = instances.count() / 5000;
      if (binSize <= 1) {
         writeInstancesTo(instances,
                          outputResource.setIsCompressed(true).getChild("part-" + id.incrementAndGet() + ".json"));
      } else {
         instances.mapToPair(i -> Tuples.$((long) Math.floor(Math.random() * binSize), i))
                  .groupByKey()
                  .forEachLocal((key, list) -> {
                                   Resource r = outputResource.setIsCompressed(true).getChild(
                                      "part-" + id.incrementAndGet() + ".json");
                                   writeInstancesTo(StreamingContext.local().stream(list), r);
                                }
                               );
      }
   }

   @Override
   public void close() {
      outputResource.delete();
   }

   @Override
   protected Dataset<T> create(@NonNull MStream<T> instances, @NonNull Encoder featureEncoder, @NonNull LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      Dataset<T> dataset = new OffHeapDataset<>(featureEncoder, labelEncoder, preprocessors);
      dataset.addAll(instances);
      return dataset;
   }

   @Override
   public DatasetType getType() {
      return DatasetType.OffHeap;
   }

   @Override
   public Dataset<T> mapSelf(@NonNull SerializableFunction<? super T, T> function) {
      outputResource.getChildren().forEach(Unchecked.consumer(r -> {
         InMemoryDataset<T> temp = new InMemoryDataset<>(getFeatureEncoder(), getLabelEncoder(), null);
         for (String line : r.readLines()) {
            temp.add(Example.fromJson(line, clazz));
         }
         temp.mapSelf(function);
         writeInstancesTo(temp.stream(), r);
      }));
      return this;
   }

   @Override
   public Dataset<T> shuffle(@NonNull Random random) {
      return create(stream().shuffle(), getFeatureEncoder().createNew(), getLabelEncoder().createNew(), null);
   }

   @Override
   public int size() {
      return size;
   }

   @Override
   public MStream<T> stream() {
      return StreamingContext.local().stream(outputResource.getChildren().stream()
                                                           .flatMap(function(r -> r.readLines().stream()))
                                                           .map(
                                                              function(line -> Cast.as(Example.fromJson(line, clazz))))
                                            );
   }

   private void writeInstancesTo(MStream<T> instances, Resource file) {
      try (BufferedWriter writer = new BufferedWriter(file.writer())) {
         for (T instance : Collect.asIterable(instances.iterator())) {
            clazz = Cast.as(instance.getClass());
            writer.write(instance.toJson());
            writer.newLine();
            size++;
         }
      } catch (IOException e) {
         throw Throwables.propagate(e);
      }
   }

}// END OF OffHeapDataset
