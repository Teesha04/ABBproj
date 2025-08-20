using Microsoft.OpenApi.Models;
using IntelliInspect.Api.Services;
using IntelliInspect.Api.Storage;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);
// Controllers + Swagger
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "IntelliInspect API", Version = "v1" });
});

builder.Services.Configure<FormOptions>(o =>
{
    o.MultipartBodyLengthLimit = 2L * 1024 * 1024 * 1024; // 2 GB
    // optional, but helps with very wide CSVs
    o.ValueCountLimit = int.MaxValue;
    o.ValueLengthLimit = int.MaxValue;
});

// Raise Kestrel request body limit
builder.WebHost.ConfigureKestrel(o =>
{
    o.Limits.MaxRequestBodySize = null;
});

// >>> REGISTER YOUR APP SERVICES HERE <<<
var storageRoot = builder.Configuration["Storage:Root"] ?? "./data";
builder.Services.AddSingleton<IStorage>(new LocalFileStorage(storageRoot));
builder.Services.AddScoped<IDatasetService, DatasetService>();
// ^^^^^ these two lines MUST be above Build()

var app = builder.Build();

app.UseSwagger();
app.UseSwaggerUI();
app.MapControllers();
app.MapGet("/", () => "IntelliInspect API is running");

app.Run();
