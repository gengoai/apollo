package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.LabelEncoder;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.guava.common.base.Throwables;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import com.davidbracewell.string.StringUtils;
import com.davidbracewell.tuple.Tuples;
import lombok.NonNull;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

import static com.davidbracewell.function.Unchecked.function;

/**
 * <p>Creates a dataset that streams examples off disk to save memory.</p>
 *
 * @param <T> the example type parameter
 * @author David B. Bracewell
 */
public class OffHeapDataset<T extends Example> extends Dataset<T> {
   private static final long serialVersionUID = 1L;
   private final AtomicLong id = new AtomicLong();
   private Resource outputResource = Resources.temporaryDirectory();
   private Class<T> clazz;
   private int size = 0;

   /**
    * Instantiates a new Off heap dataset.
    *
    * @param featureEncoder the feature encoder
    * @param labelEncoder   the label encoder
    * @param preprocessors  the preprocessors
    */
   protected OffHeapDataset(Encoder featureEncoder, LabelEncoder labelEncoder, PreprocessorList<T> preprocessors) {
      super(featureEncoder, labelEncoder, preprocessors);
      outputResource.deleteOnExit();
   }

   @Override
   protected void addAll(@NonNull MStream<T> instances) {
      //TODO: Rewrite using MultiFileWriter
      final long binSize;
      if (instances.isReusable()) {
         binSize = instances.count() / 5000;
      } else {
         binSize = -1;
      }
      if (binSize <= 1) {
         writeInstancesTo(instances,
                          outputResource.getChild("part-" + id.incrementAndGet() + ".json").setIsCompressed(true));
      } else {
         instances.mapToPair(i -> Tuples.$((long) Math.floor(Math.random() * binSize), i))
                  .groupByKey()
                  .forEachLocal((key, list) -> {
                                   Resource r = outputResource.getChild("part-" + id.incrementAndGet() + ".json")
                                                              .setIsCompressed(true);
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
      return StreamingContext.local()
                             .stream(outputResource.getChildren().parallelStream()
                                                   .flatMap(function(r -> r.lines().javaStream()))
                                                   .filter(StringUtils::isNotNullOrBlank)
                                                   .map(function(line -> Cast.as(Example.fromJson(line, clazz)))));
   }

   @Override
   public Dataset<T> copy() {
      OffHeapDataset<T> copy = Cast.as(create(getStreamingContext().empty()));
      for (Resource child : outputResource.getChildren()) {
         try {
            copy.outputResource.getChild(child.baseName())
                               .write(child.readToString());
         } catch (IOException e) {
            throw Throwables.propagate(e);
         }
      }
      copy.size = this.size;
      copy.id.set(this.id.longValue());
      copy.clazz = this.clazz;
      return copy;
   }

   private void writeInstancesTo(MStream<T> instances, Resource file) {
      try (BufferedWriter writer = new BufferedWriter(file.writer())) {
         instances.forEach(Unchecked.consumer(ii -> {
            clazz = Cast.as(ii.getClass());
            if (ii.getFeatureSpace().count() > 0) {
               writer.write(ii.toJson());
               writer.newLine();
               size++;
            }
         }));
      } catch (IOException e) {
         throw Throwables.propagate(e);
      }
   }

}// END OF OffHeapDataset
