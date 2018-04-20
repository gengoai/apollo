package com.gengoai.apollo.ml.data.source;

import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.guava.common.base.Throwables;
import com.gengoai.guava.common.collect.Iterators;
import com.gengoai.io.CSV;
import com.gengoai.io.CSVReader;
import com.gengoai.io.QuietIO;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.string.StringUtils;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * <p>Reads data from a dense CSV format where every field has a value (empty features have a 0).</p>
 *
 * @author David B. Bracewell
 */
public class DenseCSVDataSource extends DataSource<Instance> {
   private static final long serialVersionUID = 1L;
   private final CSV csvFormat;
   @Getter
   @Setter
   private int classIndex = 0;
   @Getter
   @Setter
   private String labelName = null;

   /**
    * Instantiates a new Dense CSV data source.
    *
    * @param resource  the resource containing the data
    * @param hasHeader True if the csv file has a header, False if not
    */
   public DenseCSVDataSource(@NonNull Resource resource, boolean hasHeader) {
      this(resource, CSV.builder().hasHeader(hasHeader));
   }

   /**
    * Instantiates a new Dense CSV data source. Assumes that there is no header in the file.
    *
    * @param resource the resource containing the data
    */
   public DenseCSVDataSource(@NonNull Resource resource) {
      this(resource, false);
   }


   /**
    * Instantiates a new Dense CSV data source
    *
    * @param resource the resource containing the data
    * @param format   the CSV format to use for parsing the data
    */
   public DenseCSVDataSource(@NonNull Resource resource, @NonNull CSV format) {
      super(resource);
      this.csvFormat = format;
   }

   private String getName(int index, List<String> list) {
      while (index >= list.size()) {
         list.add(Integer.toString(list.size() - 1));
      }
      return list.get(index);
   }

   private MStream<Instance> resourceToStream(Resource resource, @NonNull StreamingContext context) throws IOException {
      CSVReader reader = csvFormat.reader(resource);
      List<String> names = new ArrayList<>();
      if (reader.getHeader() != null) {
         names.addAll(reader.getHeader());
      }
      if (reader.getHeader() != null && !StringUtils.isNullOrBlank(labelName)) {
         classIndex = reader.getHeader().indexOf(labelName);
         if (classIndex == -1) {
            throw new IllegalStateException(labelName + " is not in the dataset");
         }
      }
      MStream<Instance> stream = context.stream(Iterators.transform(reader.iterator(), list -> {
         List<Feature> features = new ArrayList<>();
         for (int i = 0; i < list.size(); i++) {
            if (i != classIndex) {
               features.add(Feature.real(getName(i, names), Double.parseDouble(list.get(i))));
            }
         }
         return new Instance(features, list.get(classIndex));
      }));
      stream.onClose(() -> QuietIO.closeQuietly(reader));
      return stream;
   }

   @Override
   public MStream<Instance> stream() throws IOException {
      if (getResource().isDirectory()) {
         MStream<Instance> stream = getStreamingContext().empty();
         for (MStream<Instance> s : getResource().getChildren(true)
                                                 .stream()
                                                 .map(r -> {
                                                    try {
                                                       return resourceToStream(r, getStreamingContext());
                                                    } catch (IOException e) {
                                                       throw Throwables.propagate(e);
                                                    }
                                                 }).collect(Collectors.toList())) {
            stream = stream.union(s);
         }
      }
      return resourceToStream(getResource(), getStreamingContext());
   }

}// END OF SparseCSVDataFormat
