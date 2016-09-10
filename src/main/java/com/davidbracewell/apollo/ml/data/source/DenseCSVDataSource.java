package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.io.CSV;
import com.davidbracewell.io.CSVReader;
import com.davidbracewell.io.QuietIO;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import com.davidbracewell.string.StringUtils;
import com.google.common.base.Throwables;
import com.google.common.collect.Iterators;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public class DenseCSVDataSource extends DataSource<Instance> {

  private final CSV csvFormat;
  @Getter
  @Setter
  private int classIndex = 0;
  @Getter
  @Setter
  private String labelName = null;

  /**
   * Instantiates a new Data source.
   *
   * @param resource the resource
   */
  public DenseCSVDataSource(@NonNull Resource resource, boolean hasHeader) {
    this(resource, CSV.builder().hasHeader(hasHeader));
  }

  public DenseCSVDataSource(@NonNull Resource resource) {
    this(resource, false);
  }


  public DenseCSVDataSource(@NonNull Resource resource, @NonNull CSV format) {
    super(resource);
    this.csvFormat = format;
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

  private String getName(int index, List<String> list) {
    while (index >= list.size()) {
      list.add(Integer.toString(list.size() - 1));
    }
    return list.get(index);
  }

}// END OF SparseCSVDataFormat
