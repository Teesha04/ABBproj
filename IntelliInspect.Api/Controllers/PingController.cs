using Microsoft.AspNetCore.Mvc;

namespace IntelliInspect.Api.Controllers;

[ApiController]
[Route("api/ping")]
public class PingController : ControllerBase
{
    [HttpGet]
    public IActionResult Get() => Ok(new { message = "pong" });
}
